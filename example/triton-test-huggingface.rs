use std::error::Error;
use ndarray::{Array, Array3, Array4, arr3};
use std::vec::Vec;
use std::collections::HashMap;

use triton_rust::TritonInference;
use triton_rust::inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};

use triton_rust::cuda_shared_memory::CudaSharedMemoryRegionHandle;
use triton_rust::system_shared_memory::SystemSharedMemoryRegionHandle;

fn main() -> Result<(), Box<dyn Error>> {
    let mut triton_inferer = TritonInference::connect("http://0.0.0.0:7001").unwrap();

    /* Create a shared memory zone for exchanging data */

    let size_of_output = 8*768*4;

    triton_inferer.unregister_system_shared_memory("input_ids_data");
    triton_inferer.unregister_system_shared_memory("attention_mask_data");

    let mut system_mem_zone_ids = triton_inferer.create_system_shared_memory("input_ids_data", "/input_ids_data", 512).unwrap();
    let mut system_mem_zone_mask = triton_inferer.create_system_shared_memory("attention_mask_data", "/attention_mask_data", 512).unwrap();

    let mut system_mem_zone_output = triton_inferer.create_system_shared_memory("output_data", "/output_data", size_of_output).unwrap();

    let ids = Array::from_shape_vec((1, 8), vec![101, 8292, 6895, 9765, 4895, 3231, 999, 102]).unwrap();
    let mask = Array::from_shape_vec((1, 8), vec![1, 1, 1, 1, 1, 1, 1, 1]).unwrap();

    /* Infer using shared memory */
    let mut infer_inputs = Vec::<InferInputTensor>::with_capacity(2);
    let ids_input_params = triton_inferer.get_system_shared_memory_params("input_ids_data", 8*8, 0);
    infer_inputs.push(triton_inferer.get_infer_input("input_ids", "INT64", &[1, 8], ids_input_params));

    let mask_input_params = triton_inferer.get_system_shared_memory_params("attention_mask_data", 8*8, 0);
    infer_inputs.push(triton_inferer.get_infer_input("attention_mask", "INT64", &[1, 8], mask_input_params));

    system_mem_zone_ids.copy_array_2_int(&ids);
    system_mem_zone_mask.copy_array_2_int(&mask);

    let mut infer_outputs = Vec::<InferRequestedOutputTensor>::with_capacity(1);
    let output_params = triton_inferer.get_system_shared_memory_params("output_data", size_of_output, 0);
    infer_outputs.push(triton_inferer.get_infer_output("last_hidden_state", output_params));

    let response  = triton_inferer.infer("distilcamembert", "1", "25", infer_inputs, infer_outputs, Vec::<Vec<u8>>::new()).unwrap();

    let output_class = system_mem_zone_output.get_data(size_of_output, 0);

    let array = unsafe { output_class.align_to::<f32>().1 };
    let array_nd = Array::from_iter(array.iter());
    println!("{:?}", array_nd.into_shape((1,8,768)));

    /* Unregister shared memory zone */
    let response = triton_inferer.unregister_system_shared_memory("input_ids_data").unwrap();
    let response = triton_inferer.unregister_system_shared_memory("attention_mask_data").unwrap();
    let response = triton_inferer.unregister_system_shared_memory("output_data").unwrap();

    system_mem_zone_ids.destroy();
    system_mem_zone_mask.destroy();
    system_mem_zone_output.destroy();

    Ok(())
}