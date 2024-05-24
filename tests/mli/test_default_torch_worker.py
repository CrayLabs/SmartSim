# # BSD 2-Clause License
# #
# # Copyright (c) 2021-2024, Hewlett Packard Enterprise
# # All rights reserved.
# #
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# #    list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# #    this list of conditions and the following disclaimer in the documentation
# #    and/or other materials provided with the distribution.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# import io
# import pathlib
# import typing as t

# import pytest
# import torch

# from smartsim._core.mli.infrastructure.worker.integratedtorchworker import (
#     IntegratedTorchWorker,
# )
# import smartsim.error as sse
# from smartsim._core.mli.infrastructure import MemoryFeatureStore
# from smartsim._core.mli.infrastructure.worker.worker import (
#     ExecuteResult,
#     FetchInputResult,
#     FetchModelResult,
#     InferenceRequest,
#     TransformInputResult,
#     LoadModelResult,
# )
# from smartsim._core.utils import installed_redisai_backends

# # The tests in this file belong to the group_a group
# pytestmark = pytest.mark.group_b

# # retrieved from pytest fixtures
# is_dragon = pytest.test_launcher == "dragon"
# torch_available = "torch" in installed_redisai_backends()


# @pytest.fixture
# def persist_torch_model(test_dir: str) -> pathlib.Path:
#     test_path = pathlib.Path(test_dir)
#     model_path = test_path / "basic.pt"

#     model = torch.nn.Linear(2, 1)
#     torch.save(model, model_path)

#     return model_path


# # def test_deserialize() -> None:
# #     """Verify that serialized requests are properly deserialized to
# #     and converted to the internal representation used by ML workers"""
# #     worker = SampleTorchWorker
# #     buffer = io.BytesIO()

# #     exp_model_key = "model-key"
# #     msg = InferenceRequest(model_key=exp_model_key)
# #     pickle.dump(msg, buffer)

# #     deserialized: InferenceRequest = worker.deserialize(buffer.getvalue())

# #     assert deserialized.model_key == exp_model_key
# #     # assert deserialized.backend == exp_backend


# @pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
# def test_load_model_from_disk(persist_torch_model: pathlib.Path) -> None:
#     """Verify that a model can be loaded using a FileSystemFeatureStore"""
#     worker = IntegratedTorchWorker
#     request = InferenceRequest(raw_model=persist_torch_model.read_bytes())

#     fetch_result = FetchModelResult(persist_torch_model.read_bytes())
#     load_result = worker.load_model(request, fetch_result)

#     input = torch.randn(2)
#     pred = load_result.model(input)

#     assert pred


# @pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
# def test_transform_input() -> None:
#     """Verify that the default input transform operation is a no-op copy"""
#     rows, cols = 1, 4
#     num_values = 7
#     tensors = [torch.randn((rows, cols)) for _ in range(num_values)]

#     request = InferenceRequest()

#     inputs: t.List[bytes] = []
#     for tensor in tensors:
#         buffer = io.BytesIO()
#         torch.save(tensor, buffer)
#         inputs.append(buffer.getvalue())

#     fetch_result = FetchInputResult(inputs)
#     worker = IntegratedTorchWorker
#     result = worker.transform_input(request, fetch_result)
#     transformed: t.Collection[torch.Tensor] = result.transformed

#     assert len(transformed) == num_values

#     for output, expected in zip(transformed, tensors):
#         assert output.shape == expected.shape
#         assert output.equal(expected)

#     transformed = list(transformed)

#     original: torch.Tensor = tensors[0]
#     assert transformed[0].equal(original)

#     # verify a copy was made
#     transformed[0] = 2 * transformed[0]
#     assert transformed[0].equal(2 * original)


# @pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
# def test_execute_model(persist_torch_model: pathlib.Path) -> None:
#     """Verify that a model executes corrrectly via the worker"""

#     # put model bytes into memory
#     model_name = "test-key"
#     feature_store = MemoryFeatureStore()
#     feature_store[model_name] = persist_torch_model.read_bytes()

#     worker = IntegratedTorchWorker
#     request = InferenceRequest(model_key=model_name)
#     fetch_result = FetchModelResult(persist_torch_model.read_bytes())
#     load_result = worker.load_model(request, fetch_result)

#     value = torch.randn(2)
#     transform_result = TransformInputResult([value])

#     execute_result = worker.execute(request, load_result, transform_result)

#     assert execute_result.predictions is not None


# @pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
# def test_execute_missing_model(persist_torch_model: pathlib.Path) -> None:
#     """Verify that a executing a model with an invalid key fails cleanly"""

#     # use key that references an un-set model value
#     model_name = "test-key"
#     feature_store = MemoryFeatureStore()
#     feature_store[model_name] = persist_torch_model.read_bytes()

#     worker = IntegratedTorchWorker
#     request = InferenceRequest(input_keys=[model_name])

#     load_result = LoadModelResult(None)
#     transform_result = TransformInputResult(
#         [torch.randn(2), torch.randn(2), torch.randn(2)]
#     )

#     with pytest.raises(sse.SmartSimError) as ex:
#         worker.execute(request, load_result, transform_result)

#     assert "Model must be loaded" in ex.value.args[0]


# @pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
# def test_transform_output() -> None:
#     """Verify that the default output transform operation is a no-op copy"""
#     rows, cols = 1, 4
#     num_values = 7
#     inputs = [torch.randn((rows, cols)) for _ in range(num_values)]
#     exp_outputs = [torch.Tensor(tensor) for tensor in inputs]

#     worker = SampleTorchWorker
#     request = InferenceRequest()
#     exec_result = ExecuteResult(inputs)

#     result = worker.transform_output(request, exec_result)

#     assert len(result.outputs) == num_values

#     for output, expected in zip(result.outputs, exp_outputs):
#         assert output.shape == expected.shape
#         assert output.equal(expected)

#     transformed = list(result.outputs)

#     # verify a copy was made
#     original: torch.Tensor = inputs[0]
#     transformed[0] = 2 * transformed[0]

#     assert transformed[0].equal(2 * original)
