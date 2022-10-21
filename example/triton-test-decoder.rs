use std::error::Error;
use ndarray::{Array, Array1, Array2, Array3, Array4, arr3};
use std::vec::Vec;
use std::collections::HashMap;

use triton_rust::TritonInference;
use triton_rust::inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};

use triton_rust::cuda_shared_memory::CudaSharedMemoryRegionHandle;
use triton_rust::system_shared_memory::SystemSharedMemoryRegionHandle;
use ndarray_npy::read_npy;

fn main() -> Result<(), Box<dyn Error>> {
    let mut triton_inferer = TritonInference::connect("http://0.0.0.0:7001").unwrap();

    /* Create a shared memory zone for exchanging data */

    let size_of_encoder = 256 * 4;
    let size_of_state = 640 * 4;
    let alphabet_length = 513;

    triton_inferer.unregister_system_shared_memory("input_state_1_data");
    triton_inferer.unregister_system_shared_memory("input_state_2_data");
    triton_inferer.unregister_system_shared_memory("encoder_data");
    triton_inferer.unregister_system_shared_memory("target_length_data");
    triton_inferer.unregister_system_shared_memory("target_data");

    triton_inferer.unregister_system_shared_memory("outputs_data");
    triton_inferer.unregister_system_shared_memory("output_state_1_data");
    triton_inferer.unregister_system_shared_memory("output_state_2_data");

    let mut system_mem_zone_input_state_1 = triton_inferer.create_system_shared_memory("input_state_1_data", "/input_state_1_data", size_of_state as u64).unwrap();
    let mut system_mem_zone_input_state_2 = triton_inferer.create_system_shared_memory("input_state_2_data", "/input_state_2_data", size_of_state as u64).unwrap();
    let mut system_mem_zone_encoder = triton_inferer.create_system_shared_memory("encoder_data", "/encoder_data", size_of_encoder as u64).unwrap();
    let mut system_mem_zone_target_length = triton_inferer.create_system_shared_memory("target_length_data", "/target_length_data", (4) as u64).unwrap();
    let mut system_mem_zone_target = triton_inferer.create_system_shared_memory("target_data", "/target_data", (4) as u64).unwrap();

    let mut system_mem_zone_output = triton_inferer.create_system_shared_memory("outputs_data", "/outputs_data", (alphabet_length * 4) as u64).unwrap();
    let mut system_mem_zone_output_state_1 = triton_inferer.create_system_shared_memory("output_state_1_data", "/output_state_1_data", size_of_state as u64).unwrap();
    let mut system_mem_zone_output_state_2 = triton_inferer.create_system_shared_memory("output_state_2_data", "/output_state_2_data", size_of_state as u64).unwrap();

    let input_state_1 = Array3::<f32>::zeros((1, 1, 640));
    let input_state_2 = Array3::<f32>::zeros((1, 1, 640));
    let encoder: Array3<f32> = read_npy("encoder_vec.npy")?;

    /* Infer using shared memory */
    let mut infer_inputs = Vec::<InferInputTensor>::with_capacity(5);

    let state_1_params = triton_inferer.get_system_shared_memory_params("input_state_1_data", size_of_state, 0);
    infer_inputs.push(triton_inferer.get_infer_input("input-states-1", "FP32", &[1, 1, 640], state_1_params));

    let state_2_params = triton_inferer.get_system_shared_memory_params("input_state_2_data", size_of_state, 0);
    infer_inputs.push(triton_inferer.get_infer_input("input-states-2", "FP32", &[1, 1, 640], state_2_params));

    let encoder_input_params = triton_inferer.get_system_shared_memory_params("encoder_data", 256*4, 0);
    infer_inputs.push(triton_inferer.get_infer_input("encoder_outputs", "FP32", &[1, 256, 1], encoder_input_params));

    let target_length_params = triton_inferer.get_system_shared_memory_params("target_length_data", 4, 0);
    infer_inputs.push(triton_inferer.get_infer_input("target_length", "INT32", &[1], target_length_params));

    let target_params = triton_inferer.get_system_shared_memory_params("target_data", 4, 0);
    infer_inputs.push(triton_inferer.get_infer_input("targets", "INT32", &[1, 1], target_params));

    println!("{:?}", input_state_1);
    println!("{:?}", input_state_2);
    println!("{:?}", encoder);

    system_mem_zone_input_state_1.copy_array(&input_state_1, 0);
    system_mem_zone_input_state_2.copy_array(&input_state_2, 0);
    system_mem_zone_encoder.copy_array(&encoder, 0);
    system_mem_zone_target_length.copy_array(&Array1::<i32>::from_elem(1, 1), 0);
    system_mem_zone_target.copy_array(&Array1::<i32>::from_elem(1, 512), 0);

    let size_of_output = (alphabet_length * 4) as u64;
    let mut infer_outputs = Vec::<InferRequestedOutputTensor>::with_capacity(3);

    let outputs_params = triton_inferer.get_system_shared_memory_params("outputs_data", size_of_output, 0);
    infer_outputs.push(triton_inferer.get_infer_output("outputs", outputs_params));

    let output_state_1_params = triton_inferer.get_system_shared_memory_params("output_state_1_data", size_of_state, 0);
    infer_outputs.push(triton_inferer.get_infer_output("output-states-1", output_state_1_params));

    let output_state_2_params = triton_inferer.get_system_shared_memory_params("output_state_2_data", size_of_state, 0);
    infer_outputs.push(triton_inferer.get_infer_output("output-states-2", output_state_2_params));

    let response  = triton_inferer.infer("decoder_macarena", "1", "25", infer_inputs, infer_outputs, Vec::<Vec<u8>>::new()).unwrap();

    let outputs: Vec<f32> = system_mem_zone_output.get_data(size_of_output, 0);
    let array_outputs_nd = Array::from_iter(outputs.into_iter());

    let output_state_1: Vec<f32> = system_mem_zone_output_state_1.get_data(size_of_state, 0);
    let array_output_state_1_nd = Array::from_iter(output_state_1.into_iter()).into_shape((1, 1, 640)).unwrap();

    let output_state_2: Vec<f32> = system_mem_zone_output_state_2.get_data(size_of_state, 0);
    let array_output_state_2_nd = Array::from_iter(output_state_2.into_iter()).into_shape((1, 1, 640)).unwrap();

    println!("{:?}", array_outputs_nd);

    /* Unregister shared memory zone */
    triton_inferer.unregister_system_shared_memory("input_state_1_data");
    triton_inferer.unregister_system_shared_memory("input_state_2_data");
    triton_inferer.unregister_system_shared_memory("encoder_data");
    triton_inferer.unregister_system_shared_memory("target_length_data");
    triton_inferer.unregister_system_shared_memory("target_data");

    triton_inferer.unregister_system_shared_memory("outputs_data");
    triton_inferer.unregister_system_shared_memory("output_state_1_data");
    triton_inferer.unregister_system_shared_memory("output_state_2_data");

    Ok(())
}