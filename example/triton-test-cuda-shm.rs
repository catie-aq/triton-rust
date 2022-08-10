use std::error::Error;
use ndarray::{Array, Array3, Array4};
use std::vec::Vec;
use std::collections::HashMap;

use triton_rust::TritonInference;
use triton_rust::inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};

use triton_rust::cuda_shared_memory::CudaSharedMemoryRegionHandle;
use triton_rust::system_shared_memory::SystemSharedMemoryRegionHandle;

fn main() -> Result<(), Box<dyn Error>> {
    let mut triton_inferer = TritonInference::connect("http://0.0.0.0:7001").unwrap();

    let mut cuda_mem_zone = triton_inferer.create_cuda_shared_memory("output1_data", 4096, 0).unwrap();

    /*let response = triton_inferer.cuda_shared_memory_status("output1_data").unwrap();
    println!("{:?}", response);*/

    /*let response = triton_inferer.unregister_cuda_shared_memory("output1_data").unwrap();
    println!("{:?}", response);*/

    cuda_mem_zone.destroy();

    Ok(())
}
