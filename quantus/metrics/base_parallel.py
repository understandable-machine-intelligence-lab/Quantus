"""This module implements the base class for creating evaluation measures."""
import functools
import math
import warnings
from abc import abstractmethod
from multiprocessing import Manager, Pool, Queue, RawArray, Array, shared_memory
import queue
from multiprocessing.managers import ArrayProxy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .base import Metric, BatchedMetric
from ..helpers import utils
from ..helpers import asserts
from ..helpers.model_interface import ModelInterface
from ..helpers.normalise_func import normalise_by_negative
from ..helpers import warn_func


class PerturbationMetric(Metric):
    """
    Batched implementation inheited from base Metric class.
    """

    @asserts.attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = False,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            #perturb_kwargs: Optional[Dict[str, Any]] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict[str, Any]] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Initialise the BatchedMetric base class.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            #perturb_kwargs=perturb_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            explain_func: Optional[Callable] = None,
            explain_func_kwargs: Optional[Dict] = None,
            model_predict_kwargs: Optional[Dict] = None,
            n_steps: int = 100,
            batch_size: int = 64,
            n_workers: int = 5,
            queue_size: int = 100,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        X, Y, A, S, model = self.prepare(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

        # initialize results array
        self.last_results = np.zeros((X.shape[0], n_steps)) * np.nan

        # create generator for generating batches
        batch_generator = self.generate_perturbed_batches(
            X=X, Y=Y, A=A,
            n_steps=n_steps,
            #perturb_kwargs=self.perturb_kwargs,
            perturb_func=self.perturb_func,
            perturb_func_kwargs=self.perturb_func_kwargs,
            batch_size=batch_size,
            queue_size=queue_size,
            n_workers=n_workers,
        )
        
        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        for batch in tqdm(batch_generator, total=n_batches):
            self.process_batch(model, **batch)

        # Call post-processing
        self.postprocess()

        self.all_results.append(self.last_results)
        return self.last_results

    def generate_perturbed_batches(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            n_steps,
            #perturb_kwargs,
            perturb_func,
            perturb_func_kwargs,
            batch_size,
            queue_size,
            n_workers,
    ):
        # define queue for perturbed instances and pool to produce them
        manager = Manager()
        queue = manager.Queue(queue_size)
        pool = Pool(processes=n_workers)

        # start pool with partial function
        pool_result = pool.starmap_async(
            functools.partial(
                self.perturb_and_queue_instance,
                queue=queue,
                n_steps=n_steps,
                #perturb_kwargs=perturb_kwargs,
                perturb_func=perturb_func,
                perturb_func_kwargs=perturb_func_kwargs,
            ),
            zip(range(len(X)), X, Y, A),
        )

        # initialize batch arrays
        x_batch = np.zeros((batch_size, *X[0].shape))
        y_batch = np.zeros((batch_size, *Y[0].shape))
        a_batch = np.zeros((batch_size, *A[0].shape))
        index_batch = np.zeros((batch_size, ), dtype=int)
        step_batch = np.zeros((batch_size, ), dtype=int)

        # TODO: add progress bar
        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        for batch_id in range(n_batches):
            for row_id in range(batch_size):
                # get queue item and put data into batch arrays
                instance  = queue.get()
                index_batch[row_id] = instance['index']
                step_batch[row_id] = instance['step']
                x_batch[row_id] = instance['x']
                y_batch[row_id] = instance['y']
                a_batch[row_id] = instance['a']

                # break inner loop if last row of last batch
                if batch_id == n_batches - 1:
                    if (row_id+1) == (n_perturbed_instances % batch_size):
                        break
                
            # yield batch as dictionary
            yield {
                'indices_batch': index_batch[:row_id+1],
                'steps_batch': step_batch[:row_id+1],
                'x_batch': x_batch[:row_id+1],
                'y_batch': y_batch[:row_id+1],
                'a_batch': a_batch[:row_id+1],
            }

        # shutdown manager after last batch has been yielded
        manager.shutdown()

    @abstractmethod
    def perturb_and_queue_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            queue,
            perturb_func,
            perturb_func_kwargs,
    ):
        pass

    @abstractmethod
    def process_batch(self, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        pass

    def postprocess(self):
        pass


class PerturbationMetricSharedIn(Metric):
    """
    Batched implementation inheited from base Metric class.
    """

    @asserts.attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = False,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            #perturb_kwargs: Optional[Dict[str, Any]] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict[str, Any]] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Initialise the BatchedMetric base class.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            #perturb_kwargs=perturb_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            explain_func: Optional[Callable] = None,
            explain_func_kwargs: Optional[Dict] = None,
            model_predict_kwargs: Optional[Dict] = None,
            n_steps: int = 100,
            batch_size: int = 64,
            n_workers: int = 5,
            queue_size: int = 100,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        X, Y, A, S, model = self.prepare(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

        # initialize results array
        self.last_results = np.zeros((X.shape[0], n_steps)) * np.nan

        # create generator for generating batches
        batch_generator = self.generate_perturbed_batches(
            X=X, Y=Y, A=A,
            n_steps=n_steps,
            #perturb_kwargs=self.perturb_kwargs,
            perturb_func=self.perturb_func,
            perturb_func_kwargs=self.perturb_func_kwargs,
            batch_size=batch_size,
            queue_size=queue_size,
            n_workers=n_workers,
        )
        
        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        for batch in tqdm(batch_generator, total=n_batches):
            self.process_batch(model, **batch)

        # Call post-processing
        self.postprocess()

        self.all_results.append(self.last_results)
        return self.last_results

    def perturb_wrapper(
            self,
            index,
            X_name=None, X_shape=None, X_dtype=None,
            Y_name=None, Y_shape=None, Y_dtype=None,
            A_name=None, A_shape=None, A_dtype=None,
            queue=None,
            n_steps=None,
            perturb_func=None,
            perturb_func_kwargs=None,
            **kwargs,
    ):
        # get shared memory blocks by name
        X_shared = shared_memory.SharedMemory(name=X_name)
        Y_shared = shared_memory.SharedMemory(name=Y_name)
        A_shared = shared_memory.SharedMemory(name=A_name)

        # map shared memory to numpy array
        X_shared_np = np.ndarray(X_shape, dtype=X_dtype, buffer=X_shared.buf)
        Y_shared_np = np.ndarray(Y_shape, dtype=Y_dtype, buffer=Y_shared.buf)
        A_shared_np = np.ndarray(A_shape, dtype=A_dtype, buffer=A_shared.buf)

        self.perturb_and_queue_instance(
            index=index,
            x_instance=X_shared_np[index],
            y_instance=Y_shared_np[index],
            a_instance=A_shared_np[index],
            queue=queue,
            n_steps=n_steps,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            **kwargs,
        )

        # close connections to shared memory blocks
        X_shared.close()
        Y_shared.close()
        A_shared.close()

    def generate_perturbed_batches(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            n_steps,
            #perturb_kwargs,
            perturb_func,
            perturb_func_kwargs,
            batch_size,
            queue_size,
            n_workers,
    ):
        # define shared structures for perturbation workers
        manager = Manager()
        queue = manager.Queue(queue_size)

        # create shared memory blocks for X, Y, A
        X_shared = shared_memory.SharedMemory(create=True, size=X.nbytes)
        Y_shared = shared_memory.SharedMemory(create=True, size=Y.nbytes)
        A_shared = shared_memory.SharedMemory(create=True, size=A.nbytes)

        # map shared memory blocks to numpy arrays
        X_shared_np = np.ndarray(X.shape, dtype=X.dtype, buffer=X_shared.buf)
        Y_shared_np = np.ndarray(Y.shape, dtype=Y.dtype, buffer=Y_shared.buf)
        A_shared_np = np.ndarray(A.shape, dtype=A.dtype, buffer=A_shared.buf)

        # write data into shared memory
        X_shared_np[:] = X[:]
        Y_shared_np[:] = Y[:]
        A_shared_np[:] = A[:]
        
        # start pool with partial function
        pool = Pool(processes=n_workers)
        pool_result = pool.map_async(
            functools.partial(
                self.perturb_wrapper,
                X_name=X_shared.name, X_shape=X.shape, X_dtype=X.dtype,
                Y_name=Y_shared.name, Y_shape=Y.shape, Y_dtype=Y.dtype,
                A_name=A_shared.name, A_shape=A.shape, A_dtype=A.dtype,
                queue=queue,
                n_steps=n_steps,
                #perturb_kwargs=perturb_kwargs,
                perturb_func=perturb_func,
                perturb_func_kwargs=perturb_func_kwargs,
            ),
            range(len(X)),
        )

        # initialize batch arrays
        x_batch = np.zeros((batch_size, *X[0].shape))
        y_batch = np.zeros((batch_size, *Y[0].shape))
        a_batch = np.zeros((batch_size, *A[0].shape))
        index_batch = np.zeros((batch_size, ), dtype=int)
        step_batch = np.zeros((batch_size, ), dtype=int)

        # TODO: add progress bar
        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        for batch_id in range(n_batches):
            for row_id in range(batch_size):
                # get queue item and put data into batch arrays
                instance  = queue.get()
                index_batch[row_id] = instance['index']
                step_batch[row_id] = instance['step']
                x_batch[row_id] = instance['x']
                y_batch[row_id] = instance['y']
                a_batch[row_id] = instance['a']

                # break inner loop if last row of last batch
                if batch_id == n_batches - 1:
                    if (row_id+1) == (n_perturbed_instances % batch_size):
                        break
                
            # yield batch as dictionary
            yield {
                'indices_batch': index_batch[:row_id+1],
                'steps_batch': step_batch[:row_id+1],
                'x_batch': x_batch[:row_id+1],
                'y_batch': y_batch[:row_id+1],
                'a_batch': a_batch[:row_id+1],
            }

        # close and join pool
        pool.close()
        pool.join()

        # close connections to shared memory blocks
        X_shared.close()
        Y_shared.close()
        A_shared.close()

        # free and release shared memory blocks
        X_shared.unlink()
        Y_shared.unlink()
        A_shared.unlink()

        # shutdown manager after last batch has been yielded
        manager.shutdown()

    @abstractmethod
    def perturb_and_queue_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            queue,
            perturb_func,
            perturb_func_kwargs,
    ):
        pass

    @abstractmethod
    def process_batch(self, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        pass

    def postprocess(self):
        pass



class PerturbationMetricSharedInOut(Metric):
    """
    Batched implementation inheited from base Metric class.
    """

    @asserts.attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = False,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            #perturb_kwargs: Optional[Dict[str, Any]] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict[str, Any]] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Initialise the BatchedMetric base class.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            #perturb_kwargs=perturb_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            explain_func: Optional[Callable] = None,
            explain_func_kwargs: Optional[Dict] = None,
            model_predict_kwargs: Optional[Dict] = None,
            n_steps: int = 100,
            batch_size: int = 64,
            n_workers: int = 5,
            buffer_size: int = 128,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        X, Y, A, S, model = self.prepare(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

        # initialize results array
        self.last_results = np.zeros((X.shape[0], n_steps)) * np.nan

        # create generator for generating batches
        batch_generator = self.generate_perturbed_batches(
            X=X, Y=Y, A=A,
            n_steps=n_steps,
            #perturb_kwargs=self.perturb_kwargs,
            perturb_func=self.perturb_func,
            perturb_func_kwargs=self.perturb_func_kwargs,
            batch_size=batch_size,
            buffer_size=buffer_size,
            n_workers=n_workers,
        )
        
        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        for batch in tqdm(batch_generator, total=n_batches):
            self.process_batch(model, **batch)

        # Call post-processing
        self.postprocess()

        self.all_results.append(self.last_results)
        return self.last_results

    def perturb_wrapper(
            self,
            index,
            X_name=None, X_shape=None, X_dtype=None,
            Y_name=None, Y_shape=None, Y_dtype=None,
            A_name=None, A_shape=None, A_dtype=None,
            buf_name=None, buf_shape=None, buf_dtype=None,
            queue_free=None, queue_alloc=None,
            n_steps=None,
            perturb_func=None,
            perturb_func_kwargs=None,
            **kwargs,
    ):
        # get shared memory blocks by name
        X_shared = shared_memory.SharedMemory(name=X_name)
        Y_shared = shared_memory.SharedMemory(name=Y_name)
        A_shared = shared_memory.SharedMemory(name=A_name)
        buf_shared = shared_memory.SharedMemory(name=buf_name)

        # map shared memory to numpy array
        X_shared_np = np.ndarray(X_shape, dtype=X_dtype, buffer=X_shared.buf)
        Y_shared_np = np.ndarray(Y_shape, dtype=Y_dtype, buffer=Y_shared.buf)
        A_shared_np = np.ndarray(A_shape, dtype=A_dtype, buffer=A_shared.buf)
        buf_shared_np = np.ndarray(buf_shape, dtype=buf_dtype, buffer=buf_shared.buf)

        generator = self.perturb_and_yield_instance(
            index=index,
            x_instance=X_shared_np[index],
            y_instance=Y_shared_np[index],
            a_instance=A_shared_np[index],
            n_steps=n_steps,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            **kwargs,
        )

        for perturbed_instance in generator:
            # get free row of buffer
            buffer_row_id = queue_free.get()

            # write perturbed instance into shared buffer
            buf_shared_np[buffer_row_id] = perturbed_instance['x_perturbed']

            # put index, step and buffer_row_id into alloc queue
            queue_alloc.put(
                {
                    'index': perturbed_instance['index'],
                    'step': perturbed_instance['step'],
                    'buffer_row': buffer_row_id,
                }
            )

        # close connections to shared memory blocks
        X_shared.close()
        Y_shared.close()
        A_shared.close()
        buf_shared.close()

    def generate_perturbed_batches(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            n_steps,
            #perturb_kwargs,
            perturb_func,
            perturb_func_kwargs,
            batch_size,
            buffer_size,
            n_workers,
    ):
        # define shared structures for perturbation workers

        # create shared memory blocks for X, Y, A
        X_shared = shared_memory.SharedMemory(create=True, size=X.nbytes)
        Y_shared = shared_memory.SharedMemory(create=True, size=Y.nbytes)
        A_shared = shared_memory.SharedMemory(create=True, size=A.nbytes)

        # map shared memory blocks to numpy arrays
        X_shared_np = np.ndarray(X.shape, dtype=X.dtype, buffer=X_shared.buf)
        Y_shared_np = np.ndarray(Y.shape, dtype=Y.dtype, buffer=Y_shared.buf)
        A_shared_np = np.ndarray(A.shape, dtype=A.dtype, buffer=A_shared.buf)

        # write data into shared memory
        X_shared_np[:] = X[:]
        Y_shared_np[:] = Y[:]
        A_shared_np[:] = A[:]

        # create shared memory buffer for perturbed instances
        buffer_np = np.zeros((buffer_size, *X.shape[1:]), dtype=X.dtype)
        buffer_shared = shared_memory.SharedMemory(create=True, size=buffer_np.nbytes)
        buffer_np = np.ndarray(buffer_np.shape, dtype=buffer_np.dtype, buffer=buffer_shared.buf)
        buffer_np[:] = np.nan

        # define queues for free and filled rows 
        manager = Manager()
        queue_free = manager.Queue(buffer_size)
        queue_alloc = manager.Queue(buffer_size)
        # add all row ids of buffer to free queue
        for i in range(buffer_size):
            queue_free.put(i)

        # start pool with partial function
        pool = Pool(processes=n_workers)
        pool_result = pool.map_async(
            functools.partial(
                self.perturb_wrapper,
                X_name=X_shared.name, X_shape=X.shape, X_dtype=X.dtype,
                Y_name=Y_shared.name, Y_shape=Y.shape, Y_dtype=Y.dtype,
                A_name=A_shared.name, A_shape=A.shape, A_dtype=A.dtype,
                buf_name=buffer_shared.name, buf_shape=buffer_np.shape, buf_dtype=buffer_np.dtype,
                queue_free=queue_free, queue_alloc=queue_alloc,
                n_steps=n_steps,
                #perturb_kwargs=perturb_kwargs,
                perturb_func=perturb_func,
                perturb_func_kwargs=perturb_func_kwargs,
            ),
            range(len(X)),
        )

        # initialize batch arrays
        x_batch = np.zeros((batch_size, *X[0].shape))
        y_batch = np.zeros((batch_size, *Y[0].shape))
        a_batch = np.zeros((batch_size, *A[0].shape))
        index_batch = np.zeros((batch_size, ), dtype=int)
        step_batch = np.zeros((batch_size, ), dtype=int)

        # TODO: add progress bar
        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        for batch_id in range(n_batches):
            for output_row_id in range(batch_size):
                # get queue item and put data into batch arrays
                instance  = queue_alloc.get()
                index = instance['index']
                step = instance['step']
                buffer_row = instance['buffer_row']

                # put data into output buffer
                index_batch[output_row_id] = index
                step_batch[output_row_id] = step
                x_batch[output_row_id] = buffer_np[buffer_row]
                y_batch[output_row_id] = Y[index]
                a_batch[output_row_id] = A[index]

                # add buffer row to free queue again
                queue_free.put(buffer_row)

                # break inner loop if last row of last batch
                if batch_id == n_batches - 1:
                    if (output_row_id+1) == (n_perturbed_instances % batch_size):
                        break
                
            # yield batch as dictionary
            yield {
                'indices_batch': index_batch[:output_row_id+1].copy(),
                'steps_batch': step_batch[:output_row_id+1].copy(),
                'x_batch': x_batch[:output_row_id+1].copy(),
                'y_batch': y_batch[:output_row_id+1].copy(),
                'a_batch': a_batch[:output_row_id+1].copy(),
            }

        # close and join pool
        pool.close()
        pool.join()

        # close connections to shared memory blocks
        X_shared.close()
        Y_shared.close()
        A_shared.close()
        buffer_shared.close()

        # free and release shared memory blocks
        X_shared.unlink()
        Y_shared.unlink()
        A_shared.unlink()
        buffer_shared.unlink()

        # shutdown manager after last batch has been yielded
        manager.shutdown()

    @abstractmethod
    def perturb_and_queue_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            queue,
            perturb_func,
            perturb_func_kwargs,
    ):
        pass

    @abstractmethod
    def process_batch(self, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        pass

    def postprocess(self):
        pass


class PerturbationMetricSharedInReturnOut(Metric):
    """
    Batched implementation inheited from base Metric class.
    """

    @asserts.attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = False,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            #perturb_kwargs: Optional[Dict[str, Any]] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict[str, Any]] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Initialise the BatchedMetric base class.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            #perturb_kwargs=perturb_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            explain_func: Optional[Callable] = None,
            explain_func_kwargs: Optional[Dict] = None,
            model_predict_kwargs: Optional[Dict] = None,
            n_steps: int = 100,
            batch_size: int = 64,
            n_workers: int = 5,
            buffer_size: int = 128,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        X, Y, A, S, model = self.prepare(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

        # initialize results array
        self.last_results = np.zeros((X.shape[0], n_steps)) * np.nan

        # create generator for generating batches
        batch_generator = self.generate_perturbed_batches(
            X=X, Y=Y, A=A,
            n_steps=n_steps,
            #perturb_kwargs=self.perturb_kwargs,
            perturb_func=self.perturb_func,
            perturb_func_kwargs=self.perturb_func_kwargs,
            batch_size=batch_size,
            buffer_size=buffer_size,
            n_workers=n_workers,
        )
        
        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        for batch in tqdm(batch_generator, total=n_batches):
            self.process_batch(model, **batch)

        # Call post-processing
        self.postprocess()

        self.all_results.append(self.last_results)
        return self.last_results

    def perturb_wrapper(
            self,
            index,
            X_name=None, X_shape=None, X_dtype=None,
            Y_name=None, Y_shape=None, Y_dtype=None,
            A_name=None, A_shape=None, A_dtype=None,
            n_steps=None,
            perturb_func=None,
            perturb_func_kwargs=None,
            **kwargs,
    ):
        # get shared memory blocks by name
        X_shared = shared_memory.SharedMemory(name=X_name)
        Y_shared = shared_memory.SharedMemory(name=Y_name)
        A_shared = shared_memory.SharedMemory(name=A_name)

        # map shared memory to numpy array
        X_shared_np = np.ndarray(X_shape, dtype=X_dtype, buffer=X_shared.buf)
        Y_shared_np = np.ndarray(Y_shape, dtype=Y_dtype, buffer=Y_shared.buf)
        A_shared_np = np.ndarray(A_shape, dtype=A_dtype, buffer=A_shared.buf)

        # initialize return array
        X_perturbed = np.zeros((n_steps, *X_shape[1:]), dtype=X_dtype)

        generator = self.perturb_and_yield_instance(
            index=index,
            x_instance=X_shared_np[index],
            y_instance=Y_shared_np[index],
            a_instance=A_shared_np[index],
            n_steps=n_steps,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            **kwargs,
        )

        for step, perturbed_instance in enumerate(generator):
            X_perturbed[step] = perturbed_instance[:]

        # close connections to shared memory blocks
        X_shared.close()
        Y_shared.close()
        A_shared.close()

        return index, X_perturbed

    def generate_perturbed_batches(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            n_steps,
            #perturb_kwargs,
            perturb_func,
            perturb_func_kwargs,
            batch_size,
            buffer_size,
            n_workers,
    ):
        # define shared structures for perturbation workers

        # create shared memory blocks for X, Y, A
        X_shared = shared_memory.SharedMemory(create=True, size=X.nbytes)
        Y_shared = shared_memory.SharedMemory(create=True, size=Y.nbytes)
        A_shared = shared_memory.SharedMemory(create=True, size=A.nbytes)

        # map shared memory blocks to numpy arrays
        X_shared_np = np.ndarray(X.shape, dtype=X.dtype, buffer=X_shared.buf)
        Y_shared_np = np.ndarray(Y.shape, dtype=Y.dtype, buffer=Y_shared.buf)
        A_shared_np = np.ndarray(A.shape, dtype=A.dtype, buffer=A_shared.buf)

        # write data into shared memory
        X_shared_np[:] = X[:]
        Y_shared_np[:] = Y[:]
        A_shared_np[:] = A[:]

        # start pool with partial function
        pool = Pool(processes=n_workers)
        pool_result = pool.imap_unordered(
            functools.partial(
                self.perturb_wrapper,
                X_name=X_shared.name, X_shape=X.shape, X_dtype=X.dtype,
                Y_name=Y_shared.name, Y_shape=Y.shape, Y_dtype=Y.dtype,
                A_name=A_shared.name, A_shape=A.shape, A_dtype=A.dtype,
                n_steps=n_steps,
                #perturb_kwargs=perturb_kwargs,
                perturb_func=perturb_func,
                perturb_func_kwargs=perturb_func_kwargs,
            ),
            range(len(X)),
            chunksize=16,
        )

        # initialize batch arrays
        x_batch = np.zeros((batch_size, *X[0].shape))
        y_batch = np.zeros((batch_size, *Y[0].shape))
        a_batch = np.zeros((batch_size, *A[0].shape))
        index_batch = np.zeros((batch_size, ), dtype=int)
        step_batch = np.zeros((batch_size, ), dtype=int)

        # TODO: add progress bar
        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        output_row_id = 0
        for index, instance_perturbations in pool_result:
            for step in range(instance_perturbations.shape[0]):
                # put data into output buffer
                index_batch[output_row_id] = index
                step_batch[output_row_id] = step
                x_batch[output_row_id] = instance_perturbations[step]
                y_batch[output_row_id] = Y[index]
                a_batch[output_row_id] = A[index]

                output_row_id += 1
                if output_row_id == batch_size:
                    # yield batch as dictionary if read rows reach batch_size
                    yield {
                        'indices_batch': index_batch[:output_row_id+1].copy(),
                        'steps_batch': step_batch[:output_row_id+1].copy(),
                        'x_batch': x_batch[:output_row_id+1].copy(),
                        'y_batch': y_batch[:output_row_id+1].copy(),
                        'a_batch': a_batch[:output_row_id+1].copy(),
                    }
                    output_row_id = 0
                
        # yield last batch less rows available than batch_size
        if output_row_id != 0:
            yield {
                'indices_batch': index_batch[:output_row_id+1].copy(),
                'steps_batch': step_batch[:output_row_id+1].copy(),
                'x_batch': x_batch[:output_row_id+1].copy(),
                'y_batch': y_batch[:output_row_id+1].copy(),
                'a_batch': a_batch[:output_row_id+1].copy(),
            }

        # close and join pool
        pool.close()
        pool.join()

        # close connections to shared memory blocks
        X_shared.close()
        Y_shared.close()
        A_shared.close()

        # free and release shared memory blocks
        X_shared.unlink()
        Y_shared.unlink()
        A_shared.unlink()

    @abstractmethod
    def perturb_and_queue_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            queue,
            perturb_func,
            perturb_func_kwargs,
    ):
        pass

    @abstractmethod
    def process_batch(self, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        pass

    def postprocess(self):
        pass


    
class PerturbationMetricPassInReturnOut(Metric):
    """
    Batched implementation inheited from base Metric class.
    """

    @asserts.attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = False,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            #perturb_kwargs: Optional[Dict[str, Any]] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict[str, Any]] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Initialise the BatchedMetric base class.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            #perturb_kwargs=perturb_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            explain_func: Optional[Callable] = None,
            explain_func_kwargs: Optional[Dict] = None,
            model_predict_kwargs: Optional[Dict] = None,
            n_steps: int = 100,
            batch_size: int = 64,
            n_workers: int = 5,
            buffer_size: int = 128,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        X, Y, A, S, model = self.prepare(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

        # initialize results array
        self.last_results = np.zeros((X.shape[0], n_steps)) * np.nan

        # create generator for generating batches
        batch_generator = self.generate_perturbed_batches(
            X=X, Y=Y, A=A,
            n_steps=n_steps,
            #perturb_kwargs=self.perturb_kwargs,
            perturb_func=self.perturb_func,
            perturb_func_kwargs=self.perturb_func_kwargs,
            batch_size=batch_size,
            buffer_size=buffer_size,
            n_workers=n_workers,
        )
        
        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        for batch in tqdm(batch_generator, total=n_batches):
            self.process_batch(model, **batch)

        # Call post-processing
        self.postprocess()

        self.all_results.append(self.last_results)
        return self.last_results

    def perturb_wrapper(
            self, args,
            n_steps=None,
            perturb_func=None,
            perturb_func_kwargs=None,
            **kwargs,
    ):
        index, x_instance, y_instance, a_instance = args

        # initialize return array
        X_perturbed = np.zeros((n_steps, *x_instance.shape), dtype=x_instance.dtype)

        generator = self.perturb_and_yield_instance(
            index=index,
            x_instance=x_instance,
            y_instance=y_instance,
            a_instance=a_instance,
            n_steps=n_steps,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            **kwargs,
        )

        for step, perturbed_instance in enumerate(generator):
            X_perturbed[step] = perturbed_instance[:]

        return index, X_perturbed

    def generate_perturbed_batches(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            n_steps,
            #perturb_kwargs,
            perturb_func,
            perturb_func_kwargs,
            batch_size,
            buffer_size,
            n_workers,
    ):
        # start pool with partial function
        pool = Pool(processes=n_workers)
        pool_result = pool.imap_unordered(
            functools.partial(
                self.perturb_wrapper,
                n_steps=n_steps,
                #perturb_kwargs=perturb_kwargs,
                perturb_func=perturb_func,
                perturb_func_kwargs=perturb_func_kwargs,
            ),
            zip(range(len(X)), X, Y, A),
        )

        # initialize batch arrays
        x_batch = np.zeros((batch_size, *X[0].shape))
        y_batch = np.zeros((batch_size, *Y[0].shape))
        a_batch = np.zeros((batch_size, *A[0].shape))
        index_batch = np.zeros((batch_size, ), dtype=int)
        step_batch = np.zeros((batch_size, ), dtype=int)

        # TODO: add progress bar
        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        output_row_id = 0
        for index, instance_perturbations in pool_result:
            for step in range(instance_perturbations.shape[0]):
                # put data into output buffer
                index_batch[output_row_id] = index
                step_batch[output_row_id] = step
                x_batch[output_row_id] = instance_perturbations[step]
                y_batch[output_row_id] = Y[index]
                a_batch[output_row_id] = A[index]

                output_row_id += 1
                if output_row_id == batch_size:
                    # yield batch as dictionary if read rows reach batch_size
                    yield {
                        'indices_batch': index_batch[:output_row_id+1].copy(),
                        'steps_batch': step_batch[:output_row_id+1].copy(),
                        'x_batch': x_batch[:output_row_id+1].copy(),
                        'y_batch': y_batch[:output_row_id+1].copy(),
                        'a_batch': a_batch[:output_row_id+1].copy(),
                    }
                    output_row_id = 0
                
        # yield last batch less rows available than batch_size
        if output_row_id != 0:
            yield {
                'indices_batch': index_batch[:output_row_id+1].copy(),
                'steps_batch': step_batch[:output_row_id+1].copy(),
                'x_batch': x_batch[:output_row_id+1].copy(),
                'y_batch': y_batch[:output_row_id+1].copy(),
                'a_batch': a_batch[:output_row_id+1].copy(),
            }

    @abstractmethod
    def perturb_and_queue_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            queue,
            perturb_func,
            perturb_func_kwargs,
    ):
        pass

    @abstractmethod
    def process_batch(self, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        pass

    def postprocess(self):
        pass


import queue
from multiprocessing.pool import ThreadPool
class PerturbationMetricMultiThreadingQueue(Metric):
    """
    Batched implementation inheited from base Metric class.
    """

    @asserts.attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = False,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            #perturb_kwargs: Optional[Dict[str, Any]] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict[str, Any]] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Initialise the BatchedMetric base class.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            #perturb_kwargs=perturb_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            explain_func: Optional[Callable] = None,
            explain_func_kwargs: Optional[Dict] = None,
            model_predict_kwargs: Optional[Dict] = None,
            n_steps: int = 100,
            batch_size: int = 64,
            n_workers: int = 5,
            queue_size: int = 100,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        X, Y, A, S, model = self.prepare(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

        # initialize results array
        self.last_results = np.zeros((X.shape[0], n_steps)) * np.nan

        # create generator for generating batches
        batch_generator = self.generate_perturbed_batches(
            X=X, Y=Y, A=A,
            n_steps=n_steps,
            #perturb_kwargs=self.perturb_kwargs,
            perturb_func=self.perturb_func,
            perturb_func_kwargs=self.perturb_func_kwargs,
            batch_size=batch_size,
            queue_size=queue_size,
            n_workers=n_workers,
        )
        
        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        for batch in tqdm(batch_generator, total=n_batches):
            self.process_batch(model, **batch)

        # Call post-processing
        self.postprocess()

        self.all_results.append(self.last_results)
        return self.last_results

    def generate_perturbed_batches(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            n_steps,
            #perturb_kwargs,
            perturb_func,
            perturb_func_kwargs,
            batch_size,
            queue_size,
            n_workers,
    ):
        # define queue for perturbed instances and pool to produce them
        batch_queue = queue.Queue(queue_size)
        pool = ThreadPool(processes=n_workers)

        # start pool with partial function
        pool_result = pool.starmap_async(
            functools.partial(
                self.perturb_and_queue_instance,
                batch_queue=batch_queue,
                n_steps=n_steps,
                #perturb_kwargs=perturb_kwargs,
                perturb_func=perturb_func,
                perturb_func_kwargs=perturb_func_kwargs,
            ),
            zip(range(len(X)), X, Y, A),
        )

        # initialize batch arrays
        x_batch = np.zeros((batch_size, *X[0].shape))
        y_batch = np.zeros((batch_size, *Y[0].shape))
        a_batch = np.zeros((batch_size, *A[0].shape))
        index_batch = np.zeros((batch_size, ), dtype=int)
        step_batch = np.zeros((batch_size, ), dtype=int)

        # TODO: add progress bar
        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        for batch_id in range(n_batches):
            for row_id in range(batch_size):
                # get queue item and put data into batch arrays
                instance  = batch_queue.get()
                index_batch[row_id] = instance['index']
                step_batch[row_id] = instance['step']
                x_batch[row_id] = instance['x']
                y_batch[row_id] = instance['y']
                a_batch[row_id] = instance['a']

                # break inner loop if last row of last batch
                if batch_id == n_batches - 1:
                    if (row_id+1) == (n_perturbed_instances % batch_size):
                        break
                
            # yield batch as dictionary
            yield {
                'indices_batch': index_batch[:row_id+1],
                'steps_batch': step_batch[:row_id+1],
                'x_batch': x_batch[:row_id+1],
                'y_batch': y_batch[:row_id+1],
                'a_batch': a_batch[:row_id+1],
            }

        pool.close()
        pool.join()

    @abstractmethod
    def perturb_and_queue_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            batch_queue,
            perturb_func,
            perturb_func_kwargs,
    ):
        pass

    @abstractmethod
    def process_batch(self, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        pass

    def postprocess(self):
        pass


class PerturbationMetricMultiThreadingPassInReturnOut(Metric):
    """
    Batched implementation inheited from base Metric class.
    """

    @asserts.attributes_check
    def __init__(
            self,
            abs: bool = False,
            normalise: bool = False,
            normalise_func: Optional[Callable] = None,
            normalise_func_kwargs: Optional[Dict[str, Any]] = None,
            #perturb_kwargs: Optional[Dict[str, Any]] = None,
            perturb_func: Optional[Callable] = None,
            perturb_func_kwargs: Optional[Dict[str, Any]] = None,
            plot_func: Optional[Callable] = None,
            display_progressbar: bool = False,
            disable_warnings: bool = False,
            **kwargs):
        """
        Initialise the BatchedMetric base class.
        """
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            #perturb_kwargs=perturb_kwargs,
            perturb_func=perturb_func,
            perturb_func_kwargs=perturb_func_kwargs,
            plot_func=plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Union[np.ndarray, int],
            a_batch: Optional[np.ndarray] = None,
            s_batch: Optional[np.ndarray] = None,
            explain_func: Optional[Callable] = None,
            explain_func_kwargs: Optional[Dict] = None,
            model_predict_kwargs: Optional[Dict] = None,
            n_steps: int = 100,
            batch_size: int = 64,
            n_workers: int = 5,
            buffer_size: int = 128,
            **kwargs,
    ) -> Union[int, float, list, dict, None]:
        """
        TODO: documentation
        """
        X, Y, A, S, model = self.prepare(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            **kwargs,
        )

        # initialize results array
        self.last_results = np.zeros((X.shape[0], n_steps)) * np.nan

        # create generator for generating batches
        batch_generator = self.generate_perturbed_batches(
            X=X, Y=Y, A=A,
            n_steps=n_steps,
            #perturb_kwargs=self.perturb_kwargs,
            perturb_func=self.perturb_func,
            perturb_func_kwargs=self.perturb_func_kwargs,
            batch_size=batch_size,
            buffer_size=buffer_size,
            n_workers=n_workers,
        )
        
        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        for batch in tqdm(batch_generator, total=n_batches):
            self.process_batch(model, **batch)

        # Call post-processing
        self.postprocess()

        self.all_results.append(self.last_results)
        return self.last_results

    def generate_perturbed_batches(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            A: np.ndarray,
            n_steps,
            #perturb_kwargs,
            perturb_func,
            perturb_func_kwargs,
            batch_size,
            buffer_size,
            n_workers,
    ):
        def perturb_wrapper(
                index,
                self, 
                n_steps=None,
                perturb_func=None,
                perturb_func_kwargs=None,
                **kwargs,
        ):
            x_instance = X[index]
            y_instance = Y[index]
            a_instance = A[index]
            
            # initialize return array
            X_perturbed = np.zeros((n_steps, *x_instance.shape), dtype=x_instance.dtype)
            
            generator = self.perturb_and_yield_instance(
                index=index,
                x_instance=x_instance,
                y_instance=y_instance,
                a_instance=a_instance,
                n_steps=n_steps,
                perturb_func=perturb_func,
                perturb_func_kwargs=perturb_func_kwargs,
                **kwargs,
            )

            for step, perturbed_instance in enumerate(generator):
                X_perturbed[step] = perturbed_instance[:]

            return index, X_perturbed

        # start pool with partial function
        pool = ThreadPool(processes=n_workers)
        pool_result = pool.imap_unordered(
            functools.partial(
                perturb_wrapper,
                self=self,
                n_steps=n_steps,
                #perturb_kwargs=perturb_kwargs,
                perturb_func=perturb_func,
                perturb_func_kwargs=perturb_func_kwargs,
            ),
            range(len(X)),
        )

        # initialize batch arrays
        x_batch = np.zeros((batch_size, *X[0].shape))
        y_batch = np.zeros((batch_size, *Y[0].shape))
        a_batch = np.zeros((batch_size, *A[0].shape))
        index_batch = np.zeros((batch_size, ), dtype=int)
        step_batch = np.zeros((batch_size, ), dtype=int)

        n_perturbed_instances = (X.shape[0] * n_steps)
        n_batches = math.ceil(n_perturbed_instances / batch_size)
        output_row_id = 0
        for index, instance_perturbations in pool_result:
            for step in range(instance_perturbations.shape[0]):
                # put data into output buffer
                index_batch[output_row_id] = index
                step_batch[output_row_id] = step
                x_batch[output_row_id] = instance_perturbations[step]
                y_batch[output_row_id] = Y[index]
                a_batch[output_row_id] = A[index]

                output_row_id += 1
                if output_row_id == batch_size:
                    # yield batch as dictionary if read rows reach batch_size
                    yield {
                        'indices_batch': index_batch[:output_row_id+1].copy(),
                        'steps_batch': step_batch[:output_row_id+1].copy(),
                        'x_batch': x_batch[:output_row_id+1].copy(),
                        'y_batch': y_batch[:output_row_id+1].copy(),
                        'a_batch': a_batch[:output_row_id+1].copy(),
                    }
                    output_row_id = 0
                
        # yield last batch less rows available than batch_size
        if output_row_id != 0:
            yield {
                'indices_batch': index_batch[:output_row_id+1].copy(),
                'steps_batch': step_batch[:output_row_id+1].copy(),
                'x_batch': x_batch[:output_row_id+1].copy(),
                'y_batch': y_batch[:output_row_id+1].copy(),
                'a_batch': a_batch[:output_row_id+1].copy(),
            }

    @abstractmethod
    def perturb_and_queue_instance(
            self,
            index,
            x_instance,
            y_instance,
            a_instance,
            queue,
            perturb_func,
            perturb_func_kwargs,
    ):
        pass

    @abstractmethod
    def process_batch(self, indices_batch, steps_batch, x_batch, y_batch, a_batch):
        pass

    def postprocess(self):
        pass


