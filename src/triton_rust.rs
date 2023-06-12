#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use inference::{ServerLiveRequest, ServerReadyRequest, ModelReadyRequest};
use inference::{InferParameter, ModelInferRequest, ModelInferResponse, infer_parameter};
use inference::{ModelMetadataRequest, ModelMetadataResponse};
use inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use inference::{CudaSharedMemoryRegisterRequest, CudaSharedMemoryRegisterResponse};
use inference::{CudaSharedMemoryStatusRequest, CudaSharedMemoryStatusResponse};
use inference::{CudaSharedMemoryUnregisterResponse, CudaSharedMemoryUnregisterRequest};
use inference::{SystemSharedMemoryRegisterRequest, SystemSharedMemoryRegisterResponse};
use inference::{SystemSharedMemoryStatusRequest, SystemSharedMemoryStatusResponse};
use inference::{SystemSharedMemoryUnregisterRequest, SystemSharedMemoryUnregisterResponse};

use std;
use std::error::Error;
use std::vec::Vec;
use std::collections::HashMap;

/* Copyright CATIE, 2022-2023

b.albar@catie.fr

This software is governed by the CeCILL-B license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-B
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL-B license and that you accept its terms.*/

use std::{slice, mem};
use ndarray::{ArrayBase, Data, Dimension};

use tokio::runtime::Runtime;
use crossbeam_channel::bounded;

pub mod cuda_shared_memory;
pub mod system_shared_memory;

pub mod inference {
    tonic::include_proto!("inference");
}

pub struct TritonInference {
    rt: Runtime,
    client: GrpcInferenceServiceClient<tonic::transport::Channel>
}

impl TritonInference {
    pub fn connect(address: &'static str) -> Result<Self, Box<dyn Error>> {

        let rt  = Runtime::new().unwrap();
        let (tx, rx) = bounded(1);

        rt.block_on(async {
            let resp = GrpcInferenceServiceClient::connect(address).await;
            tx.send(resp).unwrap();
        });
        let client = rx.recv().unwrap().unwrap();

        Ok(TritonInference {
            rt: rt,
            client: client,
        })
    }

    pub fn is_server_live(&mut self) -> Result<bool,  Box<dyn Error>> {
        let request = tonic::Request::new(ServerLiveRequest {});

        let (tx, rx) = bounded(1);
        self.rt.block_on(async {
            let resp = self.client.server_live(request).await;
            tx.send(resp).unwrap();
        });

        let response = rx.recv().unwrap().unwrap();

        Ok(response.get_ref().live)
    }

    pub fn is_server_ready(&mut self) -> Result<bool,  Box<dyn Error>> {
        let request = tonic::Request::new(ServerReadyRequest {});

        let (tx, rx) = bounded(1);
        self.rt.block_on(async {
            let resp = self.client.server_ready(request).await;
            tx.send(resp).unwrap();
        });

        let response = rx.recv().unwrap().unwrap();

        Ok(response.get_ref().ready)
    }

    pub fn is_model_ready(&mut self, model_name: &str, version_number: &str) -> Result<bool,  Box<dyn Error>> {
        let request = tonic::Request::new(ModelReadyRequest {name: model_name.to_string(), version: version_number.to_string()});

        let (tx, rx) = bounded(1);
        self.rt.block_on(async {
            let resp = self.client.model_ready(request).await;
            tx.send(resp).unwrap();
        });

        let response = rx.recv().unwrap().unwrap();

        Ok(response.get_ref().ready)
    }

    pub fn get_infer_input(&mut self, input_name: &str, input_datatype: &str, tensor_shape: &[i64], parameters_map: HashMap<String, InferParameter>) -> InferInputTensor {

        InferInputTensor {
            name: input_name.to_string(),
            datatype: input_datatype.to_string(),
            shape: tensor_shape.to_vec(),
            parameters: parameters_map,
            contents: None
        }
    }

    pub fn get_infer_output(&mut self, input_name: &str, parameters_map: HashMap<String, InferParameter>) -> InferRequestedOutputTensor {

        InferRequestedOutputTensor {
            name: input_name.to_string(),
            parameters: parameters_map
        }
    }

    pub fn get_model_metadata(&mut self, model_name: &str, model_version: &str) -> Result<ModelMetadataResponse,  Box<dyn Error>> {
        let request = tonic::Request::new(ModelMetadataRequest {name: model_name.to_string(), version: model_version.to_string()});

        let (tx, rx) = bounded(1);
        self.rt.block_on(async {
            let resp = self.client.model_metadata(request).await;
            tx.send(resp).unwrap();
        });

        let response = rx.recv().unwrap().unwrap();

        Ok(response.get_ref().clone())
    }

    pub fn infer(&mut self, model_name: &str, model_version: &str, request_id: &str, inputs_vec: Vec<InferInputTensor>, outputs_vec: Vec<InferRequestedOutputTensor>, input_content: Vec<Vec<u8>>) -> Result<ModelInferResponse,  Box<dyn Error>> {

        let request = tonic::Request::new(
            ModelInferRequest {
                model_name: model_name.to_string(),
                model_version: model_version.to_string(),
                id: request_id.to_string(),
                parameters: HashMap::<String, InferParameter>::new(),
                inputs: inputs_vec,
                outputs: outputs_vec,
                raw_input_contents: input_content

            }
        );

        let (tx, rx) = bounded(1);
        self.rt.block_on(async {
            let resp = self.client.model_infer(request).await;
            tx.send(resp).unwrap();
        });

        let response = rx.recv().unwrap().unwrap();

        Ok(response.get_ref().clone())
    }

    pub fn get_input_content_from_ndarray<T: Data, D: Dimension>(&mut self, input_array: &ArrayBase<T, D>) -> Vec<u8> {

        let bytes_length = input_array.shape().iter().product::<usize>() * mem::size_of::<T::Elem>();
        let tensor_bytes = unsafe { slice::from_raw_parts(input_array.as_ptr() as *const u8, bytes_length).to_vec() };

        tensor_bytes
    }

    pub fn create_cuda_shared_memory(&mut self, name: &'static str, size: usize, device_id: i64) -> Result<cuda_shared_memory::CudaSharedMemoryRegionHandle,  Box<dyn Error>> {

        let mut cuda_handle = cuda_shared_memory::CudaSharedMemoryRegionHandle::create(name, size, device_id);
        let cuda_raw_handle = cuda_handle.get_raw_handle();

        let request = tonic::Request::new(
            CudaSharedMemoryRegisterRequest {
                name: name.to_string(),
                raw_handle: cuda_raw_handle,
                device_id: device_id,
                byte_size: (size as u64)
            }
        );

        let (tx, rx) = bounded(1);
        self.rt.block_on(async {
            let resp = self.client.cuda_shared_memory_register(request).await;
            tx.send(resp).unwrap();
        });

        let _response = rx.recv().unwrap().unwrap();

        Ok(cuda_handle)
    }

    pub fn cuda_shared_memory_status(&mut self, name: &'static str) -> Result<CudaSharedMemoryStatusResponse,  Box<dyn Error>> {

        let request = tonic::Request::new(
            CudaSharedMemoryStatusRequest {
                name: name.to_string()
            }
        );

        let (tx, rx) = bounded(1);
        self.rt.block_on(async {
            let resp = self.client.cuda_shared_memory_status(request).await;
            tx.send(resp).unwrap();
        });

        let response = rx.recv().unwrap().unwrap();

        Ok(response.get_ref().clone())
    }

    pub fn unregister_cuda_shared_memory(&mut self, name: &'static str) -> Result<CudaSharedMemoryUnregisterResponse,  Box<dyn Error>> {

        let request = tonic::Request::new(
            CudaSharedMemoryUnregisterRequest {
                name: name.to_string()
            }
        );

        let (tx, rx) = bounded(1);
        self.rt.block_on(async {
            let resp = self.client.cuda_shared_memory_unregister(request).await;
            tx.send(resp).unwrap();
        });

        let response = rx.recv().unwrap().unwrap();

        Ok(response.get_ref().clone())
    }

    pub fn create_system_shared_memory(&mut self, name: &'static str, key: &'static str, size: usize) -> Result<system_shared_memory::SystemSharedMemoryRegionHandle,  Box<dyn Error>> {

        let shm_handle = system_shared_memory::SystemSharedMemoryRegionHandle::create(name, key, size);

        let request = tonic::Request::new(
            SystemSharedMemoryRegisterRequest {
                name: name.to_string(),
                key: key.to_string(),
                offset: 0,
                byte_size: (size as u64)
            }
        );

        let (tx, rx) = bounded(1);
        self.rt.block_on(async {
            let resp = self.client.system_shared_memory_register(request).await;
            tx.send(resp).unwrap();
        });

        let _response = rx.recv().unwrap().unwrap();

        Ok(shm_handle)
    }

    pub fn system_shared_memory_status(&mut self, name: &'static str) -> Result<SystemSharedMemoryStatusResponse,  Box<dyn Error>> {

        let request = tonic::Request::new(
            SystemSharedMemoryStatusRequest {
                name: name.to_string()
            }
        );

        let (tx, rx) = bounded(1);
        self.rt.block_on(async {
            let resp = self.client.system_shared_memory_status(request).await;
            tx.send(resp).unwrap();
        });

        let response = rx.recv().unwrap().unwrap();

        Ok(response.get_ref().clone())
    }

    pub fn unregister_system_shared_memory(&mut self, name: &'static str) -> Result<SystemSharedMemoryUnregisterResponse,  Box<dyn Error>> {

        let request = tonic::Request::new(
            SystemSharedMemoryUnregisterRequest {
                name: name.to_string()
            }
        );

        let (tx, rx) = bounded(1);
        self.rt.block_on(async {
            let resp = self.client.system_shared_memory_unregister(request).await;
            tx.send(resp).unwrap();
        });

        let response = rx.recv().unwrap().unwrap();

        Ok(response.get_ref().clone())
    }

    pub fn get_system_shared_memory_params(&mut self, name: &'static str, size: u64, offset: u64) -> HashMap<String, InferParameter> {
        let params = HashMap::from([
            ("shared_memory_region".to_string(), InferParameter { parameter_choice: Some(infer_parameter::ParameterChoice::StringParam(name.to_string())) }),
            ("shared_memory_byte_size".to_string(), InferParameter { parameter_choice: Some(infer_parameter::ParameterChoice::Int64Param(size.try_into().unwrap())) }),
            ("shared_memory_offset".to_string(), InferParameter { parameter_choice: Some(infer_parameter::ParameterChoice::Int64Param(offset.try_into().unwrap())) })
        ]);

        params
    }
}
