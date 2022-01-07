use std::error::Error;
use ndarray::{Array, Array3};
use std::vec::Vec;
use std::collections::HashMap;

use triton_rust::TritonInference;
use triton_rust::inference::model_infer_request::{InferInputTensor};

use triton_rust::shared_memory::CudaSharedMemoryRegionHandle;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut triton_inferer = TritonInference::connect("http://0.0.0.0:8001").await?;

    let response = triton_inferer.is_server_ready().await?;
    println!("{:?}", response);

    let response = triton_inferer.is_model_ready("proditec-efficientnet", "1").await?;
    println!("{:?}", response);

    let response = triton_inferer.get_model_metadata("proditec-efficientnet", "1").await?;
    println!("{:?}", response);

    let rand_image = Array3::<f64>::zeros((3, 512, 512));

    let mut cuda_region = CudaSharedMemoryRegionHandle::create("image_input", 3145728, 1);
    let raw_handle = cuda_region.get_raw_handle();

    /*let mut infer_inputs = Vec::<InferInputTensor>::with_capacity(1);
    infer_inputs.push(triton_inferer.get_infer_input("image_input", "FP32", &[1, 3, 512, 512], HashMap::new()));

    let mut data_inputs = Vec::<Vec<u8>>::with_capacity(1);
    data_inputs.push(triton_inferer.get_input_content_vector(&rand_image));

    let response  = triton_inferer.infer("proditec-efficientnet", "1", "25", infer_inputs, data_inputs).await?;
    println!("{:?}", response);*/

    Ok(())
}
