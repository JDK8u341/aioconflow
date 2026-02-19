import abc
import enum
import os
import random
from aiomultiprocess import Pool
from abc import ABC
from typing import Union, Iterable, TypeVar, Callable, Any, AsyncIterable,Type
import asyncio
import inspect
import functools
import itertools
from multiprocessing import parent_process


MAX_PROCESS_COUNT = os.cpu_count()  #最大进程使用量
USE_PROCESS = 0     #已使用进程数量
USE_P_LOCK = asyncio.Lock() #已使用进程数量控制锁
BASE_WAIT_TIME = 0.1   #基础等待时长
MAX_WAIT_TIME = 0.5   #最大等待时长
MAX_CALC_RETRY_COUNT = 10 #最大计算时重试次数,防止计算时指数爆炸，浪费CPU资源

#指数退避计算
delay_time = lambda retry_count: min(BASE_WAIT_TIME * (2 ** min(retry_count,MAX_CALC_RETRY_COUNT)), MAX_WAIT_TIME) + random.random()

T = TypeVar("T")
V = TypeVar("V")

async def call_func(self,is_async: bool,has_state:bool,func,*args,**kwargs) -> V:
    # 如果是异步函数调用await
    if is_async:
        # 如果是有状态的则包含self
        if has_state:
            return await func(self,*args,**kwargs)
        return await func(*args,**kwargs)
    else:
        # 如果不是使用线程池运行
        loop_ = asyncio.get_running_loop()
        if has_state:
            func_p = functools.partial(func,self, *args, **kwargs)
        else:
            func_p = functools.partial(func, *args, **kwargs)
        return await loop_.run_in_executor(None, func_p)

class Signal(enum.Enum):
    EXIT = "exit"       #退出
    BREAK_MODEL = "break_model"     #跳出模型
    BREAK_LOOP = "break_loop" #跳出循环
    CONTINUE_LOOP = "continue_loop"   #跳过循环
    NORMAL = "normal"   #正常
    NONE = "none" #空，表示忽略本次处理，不被计入结果中
    DIRECT_ITER = "DIRECT_ITER" #表示这一份迭代器不按常规方式解析，按字面量传递
    ERROR = "error" #出错

#包装类，包装数据和信号
class DataWithSignal:
    def __init__(self,data: T,signal: Signal = Signal.NORMAL):
        self.signal = signal    #信号
        self.data = data    #数据

    #获取信号
    def get_signal(self) -> Signal:
        return self.signal

    #获取数据
    def get_data(self) -> T:
        return self.data
    #快速获取数据
    def __call__(self) -> T:
        return self.data
    #打印
    def __str__(self) -> str:
        return f"DataWithSignal<Data: {self.data}, Signal: {self.signal}>"

    #对于DIRECT_ITER的便捷数据获取处理
    def __iter__(self) -> (Iterable[T] | "DataWithSignal"):
        if self.signal == Signal.DIRECT_ITER:
            return iter(self.data)
        else:
            return self

    # 对于DIRECT_ITER的便捷处理
    def __aiter__(self) -> (AsyncIterable[T] | "DataWithSignal"):
        if self.signal == Signal.DIRECT_ITER:
            return aiter(self.data)
        else:
            return self





async def has_remaining_process(use_process_num: int,is_user_set=False) -> int:
    global USE_PROCESS
    async with USE_P_LOCK:
            if MAX_PROCESS_COUNT - USE_PROCESS >= use_process_num:  #如果能够满足资源直接分配
                USE_PROCESS += use_process_num
                return use_process_num
            elif MAX_PROCESS_COUNT - USE_PROCESS > 0 and not is_user_set:   #如果不能满足资源但是还有资源分配剩下的所有资源
                    USE_PROCESS += MAX_PROCESS_COUNT - USE_PROCESS
                    return MAX_PROCESS_COUNT - USE_PROCESS
            else:   #否则不分配
                return 0

#Handle抽象基类
class Handle(ABC):
    @abc.abstractmethod
    async def handle(self, data: "T",context_bag:"ContextBag") -> "V":
        raise NotImplementedError

    #并发处理
    async def concurrency_handle(self,data: "T",context_bag:"ContextBag") -> tuple["V","ContextBag"]:
        clean_context_bag = await context_bag.get_clear_context_bag()   #获取干净副本
        res = await self.handle(data,clean_context_bag) #使用干净副本处理
        # 如果是单层调用则手动更新
        if isinstance(self,Layer):
            await clean_context_bag.update_contexts(self,data)
        return res,clean_context_bag    #返回结果和上下文包



class Layer(Handle,ABC):   #抽象基类Layer
    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    async def handle(self, data: "T",context_bag:"ContextBag") -> "V":   #每个层自己的处理方法
        raise NotImplementedError

#上下文包
class ContextBag:
    def __init__(self,*contexts: "Context"):
        self.contexts = {context.CONTEXT_TYPE_NAME:context for context in contexts} #携带的上下文

    #工厂方法：创建上下文包
    @classmethod
    async def create(cls,*contexts: "Context") -> "ContextBag":
        context_bag = cls(*contexts)    #创建对象
        #遍历每个上下文对象分别初始化
        for con in context_bag.contexts.values():
            await con.init_context(context_bag)
        return context_bag  #返回新的上下文包

    #合并上下文对象
    async def merge_context(self,context_obj,concurrency_merge=False):
        if context_obj.CONTEXT_TYPE_NAME in self.contexts.keys():   #如果该类型已经存在
            # 则调用合并方法
            # 检查合并方式
            if concurrency_merge:
                await self.contexts[context_obj.CONTEXT_TYPE_NAME].concurrency_merge(context_obj)
            else:
                await self.contexts[context_obj.CONTEXT_TYPE_NAME].direct_merge(context_obj)
        else:
            self.contexts[context_obj.CONTEXT_TYPE_NAME] = context_obj     #否则添加

    #更新上下文
    async def update_contexts(self,now_layer: "Layer",data: T):
        #遍历更新
        for con in self.contexts.values():
            await con.update(self,now_layer,data)

    # 获取干净副本
    async def get_clear_context_bag(self) -> "ContextBag":
        #遍历获取每个上下文的干净副本
        return await ContextBag.create(*[context.copy() for context in self.contexts.values()])

    #与另一个上下文包合并
    async def merge(self,contexts_bag: "ContextBag",concurrency_merge=False):
        for name,context in contexts_bag.contexts.items():
            await self.merge_context(context,concurrency_merge=concurrency_merge)

    #获取上下文
    def get_context(self,context_name):
        return self.contexts[context_name]


#上下文对象
class Context(ABC):
    #上下文类型名称
    CONTEXT_TYPE_NAME = "BaseContext"

    #创建实例方法
    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    #初始化钩子
    @abc.abstractmethod
    async def init_context(self,context_bag:"ContextBag") -> "Context":
        raise NotImplementedError

    #每层更新钩子
    @abc.abstractmethod
    async def update(self,context_bag:"ContextBag",now_layer: Layer,data: "T"):
        raise NotImplementedError

    # 合并方法，用于与另一个同类型上下文并发合并
    @abc.abstractmethod
    async def concurrency_merge(self,context: "Context"):
        raise NotImplementedError

    #合并方法，用于与另一个同类型上下文直接合并
    @abc.abstractmethod
    async def direct_merge(self,context:"Context") -> "T":
        raise NotImplementedError

    #复制方法，用于获取干净副本
    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

    #设置每个子类的CONTEXT_TYPE_NAME
    def __init_subclass__(cls, **kwargs):
        if cls.CONTEXT_TYPE_NAME == "BaseContext":
            cls.CONTEXT_TYPE_NAME = cls.__name__

#工具上下文：LayerCnt，层计数
class LayerCnt(Context):
    CONTEXT_TYPE_NAME = "LayerCnt"
    def __init__(self):
        self.cnt = 0    #总计数
        self.merge_cnt = 0  #合并计数

    async def init_context(self,context_bag:"ContextBag"):
        #无流程
        return None

    async def update(self,context_bag:"ContextBag",now_layer: Layer,data: "T"):
        last_cnt = self.cnt #记录上一次计数
        self.cnt += 1   #增加1层经过计数
        self.cnt += self.merge_cnt  #合并merge_cnt
        print(f"UPDATE CNT {last_cnt} -> {self.cnt} on {now_layer}")
        self.merge_cnt = 0  #清空merge_cnt

    async def concurrency_merge(self,context:"LayerCnt"):
        print(f"CONCURRENCY MERGE CNT {self.merge_cnt} -> {self.merge_cnt + context.get_cnt() - self.cnt} (M: {context.get_cnt()},S: {self.cnt})")
        self.merge_cnt += (context.get_cnt() - self.cnt)    #计算增量

    async def direct_merge(self, context: "LayerCnt"):
        print(f"UPDATE MERGE CNT {self.cnt} -> {self.cnt + context.get_cnt()} (M: {context.get_cnt()},S: {self.cnt})")
        self.cnt += context.get_cnt()


    #复制方法
    def copy(self):
        new = self.__class__()
        new.cnt = self.cnt
        return new

    #自定义方法，获取计数
    def get_cnt(self):
        return self.cnt

#抽象基类：并发层
class ConcurrencyLayer(Layer,ABC):
    #抽象方法：处理
    @abc.abstractmethod
    async def handle(self, data:"T",context_bag:"ContextBag") -> "V":
        raise NotImplementedError

    #静态方法: 收集并合并数据
    @staticmethod
    async def merge_and_collect_data(result,context_bag:"ContextBag"):
        res_datas = []  #数据列表
        for i in result:    #遍历结果
            #如果出现异常直接抛出
            if isinstance(i,Exception):
                raise i
            #否则处理数据和上下文
            else:
                #解包数据
                res_data, _context_bag = i
                #添加到数据列表
                res_datas.append(res_data)
                #合并上下文
                await context_bag.merge(_context_bag,concurrency_merge=True)
        return res_datas


# 重新包装包含特殊控制流信号的列表
# 在并发中：
# - 任何一个任务退出，整个退出
# - 没有一个任务退出，但有一个任务跳出循环，整个跳出循环
#注：循环Layer的要求，所以BREAK_LOOP至少需要跳出一层模型，
#   且即使使用Layer作为循环，但是循环本身也是一个分支，所以
#   包含BREAK_MODEL，且因为会跳出循环，所以高于CONTINUE
# - 没有一个任务跳出循环,但有一个任务跳过循环，整跳过循环
#注：跳过循环需要一直跳出到循环的那一层模型，可能跳过多层模型，
#   所以包含BREAK_MODEL,且如果只有循环所包裹的一层模型，
#   也应该使用BREAK_LOOP来跳出循环，且对于Loop处理BREAK_LOOP和BREAK_MODE的方式相同
# - 没有一个任务跳过循环，但有一个任务跳出模型，整个跳出模型
# - 否则正常继续
# - 对于NONE信号则不加入结果列表
def remake_concurrency_signal(results):
    is_exit = False             #包含退出信号
    is_break_model = False      #包含跳出模型信号
    is_break_loop = False       #包含跳出循环信号
    is_continue_loop = False    #包含跳过循环信号
    _results = []   #新返回列表
    for result in results:  #遍历检查
        if isinstance(result, DataWithSignal):
            signal = result.get_signal()
            if signal == Signal.EXIT:
                is_exit = True
                _results.append(result.get_data())  #加入数据
                continue
            elif signal == Signal.BREAK_MODEL:
                is_break_model = True
                _results.append(result.get_data())
                continue
            elif signal == Signal.BREAK_LOOP:
                is_break_loop = True
                _results.append(result.get_data())
                continue
            elif signal == Signal.CONTINUE_LOOP:
                is_continue_loop = True
                _results.append(result.get_data())
                continue
            elif signal == Signal.NONE:
                continue
        _results.append(result)

    #返回包装
    if is_exit:
        return DataWithSignal(_results, Signal.EXIT)
    elif is_break_loop:
        return DataWithSignal(_results, Signal.BREAK_LOOP)
    elif is_continue_loop:
        return DataWithSignal(_results, Signal.CONTINUE_LOOP)
    elif is_break_model:
        return DataWithSignal(_results, Signal.BREAK_MODEL)
    else:
        return _results

#工具Layer：并发
class ApplyConcurrencyLayer(ConcurrencyLayer):
    def __init__(self, layer_or_model_s: Union[tuple[Layer, ...], tuple["Model", ...]], is_cpu_dense,use_process):
        self.layer_or_model_s: Union[tuple[Layer, ...], tuple["Model", ...]] = layer_or_model_s   #层和模型的列表
        self.is_cpu_dense: bool = is_cpu_dense  #是否CPU密集
        self.use_process = use_process

    async def handle(self, data: "T",context_bag:"ContextBag") -> "V":
        global USE_PROCESS

        if len(self.layer_or_model_s) == 0: #如果是空的则直接返回
            return []

        tasks = []  #任务列表
        if self.is_cpu_dense and parent_process() is None:   #如果是CPU密集型且不是sub的
            use = self.use_process if self.use_process is not None else len(self.layer_or_model_s)  # 需要使用的进程数量
            retry_count = 0
            while True:
                num = await has_remaining_process(use, is_user_set=self.use_process is not None)
                if num > 0:
                    # 创建进程池，数量为任务数与最大进程数取小的那一个
                    async with Pool(processes=num) as p:
                        #遍历层和模型的列表
                        for l in self.layer_or_model_s:
                            #上报处理
                            task = p.apply(l.concurrency_handle, (data,context_bag))
                            tasks.append(task)
                        try:
                            result = await asyncio.gather(*tasks,return_exceptions=True)
                        finally:
                            async with USE_P_LOCK:
                                USE_PROCESS -= num  #恢复使用进程数量
                    break
                else:
                    retry_count += 1
                    await asyncio.sleep(delay=delay_time(retry_count))
        else:
            result = await asyncio.gather(*(l.concurrency_handle(data,context_bag) for l in self.layer_or_model_s),return_exceptions=True)

        #合并上下文并收集结果
        res_datas = await self.merge_and_collect_data(result, context_bag)
        return remake_concurrency_signal(res_datas)


#工具Layer：映射并发
class MapConcurrencyLayer(ConcurrencyLayer):
    def __init__(self, layer_or_model: Union[Layer,"Model"],is_cpu_dense,use_process):
        self.layer_or_model = layer_or_model    #层或模型
        self.is_cpu_dense: bool = is_cpu_dense  #是否CPU密集
        self.use_process = use_process

    async def handle(self, data: T,context_bag:"ContextBag") -> V:
        global USE_PROCESS

        if ((not isinstance(data,Iterable)) or isinstance(data,(str,bytes))
                or (isinstance(data, DataWithSignal)
                    and (data.get_signal() == Signal.DIRECT_ITER
                         or data.get_signal() == Signal.ERROR))):   #如果不是可迭代或者需要按字面解析的迭代器或者是错误信号则手动包裹
            data = [data]

        if len(data) == 0:  #如果是空的则直接返回
            return []

        if self.is_cpu_dense and parent_process() is None:
            use = self.use_process if self.use_process is not None else len(data)   #需要的进程数
            retry_count = 0 #尝试次数
            while True:
                num = await has_remaining_process(use, is_user_set=self.use_process is not None)
                if num > 0:
                    try:
                        async with Pool(processes=num) as p:
                            results = await p.starmap(self.layer_or_model.concurrency_handle, iter(zip(data,itertools.repeat(context_bag, len(data))))) #批量提交

                    finally:
                        async with USE_P_LOCK:
                            USE_PROCESS -= num
                    break
                else:
                    retry_count += 1
                    await asyncio.sleep(delay=delay_time(retry_count))

        else:
            results = await asyncio.gather(*(self.layer_or_model.concurrency_handle(d,context_bag) for d in data),return_exceptions=True)

        res_datas = await self.merge_and_collect_data(results,context_bag)

        return remake_concurrency_signal(res_datas)



#特殊Layer：重试
class RetryLayer(Layer):
    def __init__(self,retry_num: int,catch_err_type: type[Exception],try_handle: Union[Layer, "Model"],fatal_handle: Union[Layer, "Model"] = None):
        self.retry_num: int = retry_num #重试次数
        self.catch_err_type: type[Exception] = catch_err_type   #抓取的Err类型
        self.try_handle = try_handle  #尝试的层或模型
        self.fatal_handle = fatal_handle    #如果尝试都失败以后的分支

    async def handle(self, data,context_bag:"ContextBag"):
        first_data = data
        retry_num = self.retry_num
        while True:
            try:
                data = await self.try_handle.handle(first_data,context_bag)
                retry_num = 0
                break
            except self.catch_err_type as e:
                retry_num -= 1
                if retry_num <= 0:
                    if self.fatal_handle is not None:
                        return await self.fatal_handle.handle(DataWithSignal((first_data,e),Signal.ERROR),context_bag)
                    else:
                        return DataWithSignal((first_data,e),Signal.ERROR)
            await asyncio.sleep(delay_time(self.retry_num-retry_num))
        return data


#特殊Layer：选择
class ChoiceLayer(Layer, ABC):
    def __init__(self, *choices: "Model"):
        self.choices:dict[str,"Model"] = {m.get_name():m for m in choices}

    #抽象方法choice，必须由子层实现
    @abc.abstractmethod
    async def choice(self, data: T,context_bag:"ContextBag") -> "Model":
        raise NotImplementedError

    #处理
    async def handle(self, data: T,context_bag:"ContextBag"):
        choice_ = await self.choice(data,context_bag)    #调用选择方法
        return await choice_.run(data,context_bag)

#特殊Layer：循环
class LoopLayer(Layer, ABC):
    def __init__(self, loops: Union["Model",Layer]):
        self.loops = loops  #循环体

    @abc.abstractmethod
    async def handle(self, data: T,context_bag:"ContextBag") -> V:
        raise NotImplementedError

    #处理循环体调用
    async def handle_call(self, data: T,context_bag:ContextBag,i = None) -> tuple[V,bool]:
        new_data = await self.loops.handle(DataWithSignal((data, i), Signal.DIRECT_ITER),context_bag) if i is not None else await self.loops.handle(data,context_bag)
        if isinstance(new_data, DataWithSignal):
            #对于EXIT信号直接退出（传出信号）
            if new_data.signal == Signal.EXIT:
                return new_data,True
            #对于BREAK_LOOP，BREAK_MODEL，取出数据再退出（不传播信号）
            elif new_data.signal == Signal.BREAK_LOOP or new_data.signal == Signal.BREAK_MODEL:
                return new_data.get_data(),True
            #对于CONTINUE_LOOP，取出数据
            elif new_data.signal == Signal.CONTINUE_LOOP:
                new_data = new_data.get_data()
            #对于NONE信号，处理作废
            elif new_data.signal == Signal.NONE:
                new_data = data
        #返回处理结果
        return new_data,False

#特殊Layer: While循环
class WhileLoopLayer(LoopLayer, ABC):
    def __init__(self, loops: Union["Model",Layer]):
        super().__init__(loops)

    @abc.abstractmethod
    async def do_while(self, data: T,context_bag:"ContextBag") -> bool:
        raise NotImplementedError

    async def handle(self, data: T,context_bag:"ContextBag") -> V:
        while await self.do_while(data,context_bag):
            #处理调用
            data,is_ret = await self.handle_call(data,context_bag)
            #如果需要返回则直接返回
            if is_ret:
                return data
        return data

#工具Layer：Iter循环
class IterLoopLayer(LoopLayer):
    def __init__(self,_iter: Iterable | AsyncIterable,loops: Union["Model",Layer]):
        super().__init__(loops)
        self._iter = _iter

    def do_while(self, data: T) -> bool:
        pass

    async def handle(self, data: T,context_bag:"ContextBag") -> V:
        if isinstance(self._iter,AsyncIterable):
            #对于异步迭代器使用异步迭代
            async for i in self._iter:
                #调用处理函数
                data,is_ret = await self.handle_call(data,context_bag,i)
                if is_ret:
                    return data
        else:
            #否则使用普通迭代
            for i in self._iter:
                data, is_ret = await self.handle_call(data, context_bag,i)
                if is_ret:
                    return data
        return data

#工具Layer：ReDo循环
class ReDoLoopLayer(LoopLayer):
    def __init__(self, loops: Union["Model",Layer],loop_num):
        super().__init__(loops)
        self.do_num = loop_num

    async def handle(self, data: T,context_bag:"ContextBag") -> V:
        cnt = 0 #计数器
        #如果计数器比目标次数小
        while cnt < self.do_num:
            #重复调用
            data, is_ret = await self.handle_call(data,context_bag)
            #如果需要返回直接反会
            if is_ret:
                return data
            #增加计数器
            cnt += 1
        return data

#工具Layer：简单While循环
class SimpleWhileLoopLayer(WhileLoopLayer):
    def __init__(self, loops: Union["Model",Layer],do_while_func):
        super().__init__(loops)
        self.do_while_func = do_while_func  #循环判断函数
        self.is_func_async = inspect.iscoroutinefunction(do_while_func) #是否为异步函数

    async def do_while(self, data: T,context_bag:"ContextBag") -> bool:
        #返回调用结果
        return await call_func(self,self.is_func_async,False,self.do_while_func,data,context_bag)

#工具Layer：退出
class ExitLayer(Layer):
    def __init__(self):
        pass
    async def handle(self, data: T,context_bag:"ContextBag") -> T:
        return DataWithSignal(data,Signal.EXIT)

#工具Layer：跳出模型
class BreakModelLayer(Layer):
    def __init__(self):
        pass
    async def handle(self, data: T,context_bag:"ContextBag") -> T:
        return DataWithSignal(data, Signal.BREAK_MODEL)

#工具Layer：函数包装
class SimpleFuncLayer(Layer):
    def __init__(self,func: Callable[[T,ContextBag], V],ret_data_self=False):
        self.func = func
        self.is_func_async = inspect.iscoroutinefunction(func)
        self.ret_data_self = ret_data_self

    async def handle(self, data: T,context_bag:"ContextBag") -> T:
        if not self.ret_data_self:
            return await call_func(self,self.is_func_async,False,self.func,data,context_bag)
        else:
            await call_func(self, self.is_func_async, False, self.func, data,context_bag)
            return data


class Model(Handle):  # 模型？？？？
    def __init__(self, name: str,context_clss: Iterable[Type[Context]] | None = None):
        self.name: str = name   #名称
        self.handles: list[Layer | Model] = []   #层合集
        self.context_clss = context_clss if context_clss is not None else []    #初始上下文

    #添加一个层
    def layer(self, layer_: Layer) -> "Model":
        self.handles.append(layer_)
        return self

    #添加一个子模型
    def model(self, model_: "Model") -> "Model":
        self.handles.append(model_)
        return self

    #添加一个简单func层
    def func_layer(self, func: Callable[[T,ContextBag], V],return_data_self=False) -> "Model":
        self.handles.append(SimpleFuncLayer(func,ret_data_self=return_data_self))
        return self

    #添加一个Apply并发
    def apply_concurrency(self, *layer_or_model_s: Union[Layer, "Model"], cpu_dense=False, use_process=None) -> "Model":
        if cpu_dense and use_process is not None and use_process < 1:
            raise ValueError("use_process must be >= 1")
        self.handles.append(ApplyConcurrencyLayer(layer_or_model_s, is_cpu_dense=cpu_dense, use_process=use_process))
        return self

    #添加一个Map并发
    def map_concurrency(self, layer_or_model: Union[Layer, "Model"], cpu_dense=False, use_process=None) -> "Model":
        if cpu_dense and use_process is not None and use_process < 1:
            raise ValueError("use_process must be >= 1")
        self.handles.append(MapConcurrencyLayer(layer_or_model, is_cpu_dense=cpu_dense, use_process=use_process))
        return self

    #添加一个for循环
    def redo_loop(self, loops: "Model",loop_num: int) -> "Model":
        self.handles.append(ReDoLoopLayer(loops, loop_num))
        return self

    #添加一个简单while循环
    def while_loop(self, loops: "Model",do_while:Callable[[T,ContextBag],bool]) -> "Model":
        self.handles.append(SimpleWhileLoopLayer(loops, do_while))
        return self

    #添加一个迭代器循环
    def iter_loop(self, loops: "Model",_iter:Iterable | AsyncIterable) -> "Model":
        self.handles.append(IterLoopLayer(_iter, loops))
        return self

    #添加一个终止
    def exit(self) -> "Model":
        self.handles.append(ExitLayer())
        return self

    #添加一个跳出
    def break_model(self) -> "Model":
        self.handles.append(BreakModelLayer())
        return self

    #运行方法
    async def run(self, data: T = None,context_bag: ContextBag = None) -> V:
        new_data = data
        context_bag = context_bag \
            if context_bag is not None \
            else await ContextBag.create(*(context_cls() for context_cls in self.context_clss)) #初始化上下文包
        for l in self.handles:
            new_data = await l.handle(data,context_bag) #运行
            await context_bag.update_contexts(l,new_data)   #更新上下文
            #检查信号
            if isinstance(new_data, DataWithSignal):
                #对于EXIT，BREAK_LOOP,CONTINUE_LOOP信号返回DataWithSignal
                if new_data.get_signal() == Signal.EXIT or new_data.get_signal() == Signal.BREAK_LOOP or new_data.get_signal() == Signal.CONTINUE_LOOP:
                    return new_data
                #对于BREAK_MODEL信号返回data
                elif new_data.get_signal() == Signal.BREAK_MODEL:
                    return new_data.get_data()
                #对于NONE信号，忽略其处理
                elif new_data.get_signal() == Signal.NONE:
                    new_data = data
            data = new_data
        return new_data

    #获取名称
    def get_name(self) -> str:
        return self.name

    #设置名称
    def set_name(self, name: str) -> "Model":
        self.name = name
        return self

    async def handle(self, data: T,context_bag:ContextBag | None = None) -> V:
        return await self.run(data,context_bag=context_bag)

    def __str__(self) -> str:
        return f"Model<name = {self.name}>"


class DecoratorLayer(Layer):
        def __init__(self,func,has_state,*args,**kwargs):  #函数参数
            self.args = args
            self.kwargs = kwargs
            self.has_state = has_state
            self.func = func
            self.is_async = inspect.iscoroutinefunction(func)    #是否为异步函数
        async def handle(self, data,context_bag:"ContextBag"):
            return await call_func(self,self.is_async, self.has_state, self.func,*(data,context_bag, *self.args), **self.kwargs)

#layer装饰器动态建类
def layer(func=None, *, has_state=False):
    # 当 @layer 被调用时（无括号），func 就是被装饰的函数
    if func is not None:
        # 直接装饰函数，使用默认的 has_state=False
        return _create_layer_decorator(func, has_state=False)

    # 当 @layer() 或 @layer(has_state=True) 被调用时
    def decorator(f):
        return _create_layer_decorator(f, has_state)

    return decorator
#创建函数
def _create_layer_decorator(func: Union[Callable[[T,ContextBag,Any],V],Callable[["DecoratorLayer", T,ContextBag, Any],V]], has_state=False):

        def wrapper(*args, **kwargs):
                #返回这个动态建立的类
                return DecoratorLayer(func,has_state,*args, **kwargs)
        return wrapper

class DecoratorChoiceLayer(ChoiceLayer):
        def __init__(self,func,has_state,*args,**kwargs):  #函数参数
            super().__init__()
            self.args = args
            self.kwargs = kwargs
            self.has_state = has_state
            self.func = func
            self.is_async = inspect.iscoroutinefunction(func)
        async def choice(self, data: T,context_bag:"ContextBag"):
            return await call_func(self,self.is_async,self.has_state,self.func,*(data,context_bag,self.choices, *self.args),**self.kwargs)
        #设置分支模型
        def set_choices(self,*choices: Model):
            self.choices = {model.get_name():model for model in choices}
            return self

        def __call__(self, *choices: Model):
            return self.set_choices(*choices)

#choice装饰器动态建类
def choice(func=None, *, has_state=False):
    if func is not None:
        # 直接装饰函数，使用默认的 has_state=False
        return _create_choice_decorator(func, has_state=False)

    def decorator(f):
        return _create_choice_decorator(f, has_state)

    return decorator

def _create_choice_decorator(func: Union[Callable[[T,ContextBag,dict[str,Model],Any],V],Callable[["DecoratorChoiceLayer", T,ContextBag, dict[str,Model], Any],V]], has_state=False):
        def wrapper(*args, **kwargs):
            #返回这个动态建立的类
            return DecoratorChoiceLayer(func,has_state,*args, **kwargs)
        return wrapper

#while_loop_layer装饰器动态建类
def while_loop_layer(func=None, *, has_state=False):
    if func is not None:
        # 直接装饰函数，使用默认的 has_state=False
        return _create_while_loop_layer_decorator(func, has_state=False)

    def decorator(f):
        return _create_while_loop_layer_decorator(f, has_state)

    return decorator


class DecoratorWhileLoopLayer(WhileLoopLayer):
    def __init__(self,func,has_state, *args, **kwargs):  # 函数参数
        super().__init__(None)
        self.args = args
        self.kwargs = kwargs
        self.func = func
        self.is_async = inspect.iscoroutinefunction(func)
        self.has_state = has_state

    async def do_while(self, data,context_bag:"ContextBag"):
        return await call_func(self, self.is_async, self.has_state, self.func, *(data,context_bag, *self.args), **self.kwargs)

    def set_loop_model(self, loops: "Model") -> WhileLoopLayer:
        self.loops = loops
        return self

    def __call__(self, loops: "Model") -> WhileLoopLayer:
        return self.set_loop_model(loops)

#创建函数
def _create_while_loop_layer_decorator(func: Union[Callable[[T,ContextBag,Any],V],Callable[["DecoratorLayer", T,ContextBag, Any],V]], has_state=False):
        def wrapper(*args, **kwargs):
                #返回这个动态建立的类
                return DecoratorWhileLoopLayer(func,has_state,*args, **kwargs)
        return wrapper