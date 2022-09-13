use std::error::Error;
use ndarray::{Array, Array3, Array4, Ix4, RawData};
use std::vec::Vec;
use std::collections::HashMap;

use triton_rust::TritonInference;
use triton_rust::inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};

use triton_rust::cuda_shared_memory::CudaSharedMemoryRegionHandle;
use triton_rust::system_shared_memory::SystemSharedMemoryRegionHandle;

fn main() -> Result<(), Box<dyn Error>> {
    let mut triton_inferer = TritonInference::connect("http://0.0.0.0:7001").unwrap();

    let response = triton_inferer.is_server_ready().unwrap();
    println!("{:?}", response);

    let response = triton_inferer.is_model_ready("proditec-efficientnet", "1").unwrap();
    println!("{:?}", response);

    let response = triton_inferer.get_model_metadata("proditec-efficientnet", "1").unwrap();
    println!("{:?}", response);

    let rand_image = Array4::<f32>::ones((1, 3, 512, 512));

    /*let mut cuda_mem_zone = triton_inferer.create_cuda_shared_memory("output1_data", 4096, 0).await?;

    let response = triton_inferer.cuda_shared_memory_status("output0_data").await?;
    println!("{:?}", response);

    let response = triton_inferer.unregister_cuda_shared_memory("output0_data").await?;
    println!("{:?}", response);

    cuda_mem_zone.destroy();*/

    /* Create a shared memory zone for exchanging data */

    triton_inferer.unregister_system_shared_memory("image_input_data");
    triton_inferer.unregister_system_shared_memory("output_data");

    let size_of_image_input: u64 = 3 * 512 * 512 * 4;
    let size_of_classes_output: u64 = 4 * 6;
    let size_of_reg_output: u64 = 4 ;

    let mut system_mem_zone_input = triton_inferer.create_system_shared_memory("image_input_data", "/image_input_simple", size_of_image_input).unwrap();
    let mut system_mem_zone_output = triton_inferer.create_system_shared_memory("output_data", "/output_simple", size_of_classes_output + size_of_reg_output).unwrap();

    let response = triton_inferer.system_shared_memory_status("image_input_data").unwrap();
    println!("{:?}", response);
    let response = triton_inferer.system_shared_memory_status("output_data").unwrap();
    println!("{:?}", response);

    /* Infer using direct communication */

    let mut infer_inputs = Vec::<InferInputTensor>::with_capacity(1);
    infer_inputs.push(triton_inferer.get_infer_input("image_input", "FP32", &[1, 3, 512, 512], HashMap::new()));

    let mut data_inputs = Vec::<Vec<u8>>::with_capacity(1);
    data_inputs.push(triton_inferer.get_input_content_vector(&rand_image));

    let response  = triton_inferer.infer("proditec-efficientnet", "1", "25", infer_inputs, Vec::<InferRequestedOutputTensor>::new(), data_inputs).unwrap();
    println!("{:?}", response);

    let output_class = unsafe { response.raw_output_contents[0].align_to::<f32>().1 };
    let reg_class = unsafe { response.raw_output_contents[1].align_to::<f32>().1 };
    println!("{:?}", output_class);
    println!("{:?}", reg_class);

    /* Infer using shared memory */
    let mut infer_inputs = Vec::<InferInputTensor>::with_capacity(1);
    let image_input_params = triton_inferer.get_system_shared_memory_params("image_input_data", size_of_image_input, 0);
    infer_inputs.push(triton_inferer.get_infer_input("image_input", "FP32", &[1, 3, 512, 512], image_input_params));

    system_mem_zone_input.copy_array(&rand_image, 0);

    let mut infer_outputs = Vec::<InferRequestedOutputTensor>::with_capacity(2);
    let class_output_params = triton_inferer.get_system_shared_memory_params("output_data", size_of_classes_output, 0);
    let reg_output_params = triton_inferer.get_system_shared_memory_params("output_data", size_of_reg_output, size_of_classes_output);
    infer_outputs.push(triton_inferer.get_infer_output("class_output", class_output_params));
    infer_outputs.push(triton_inferer.get_infer_output("reg_output", reg_output_params));

    let response  = triton_inferer.infer("proditec-efficientnet", "1", "25", infer_inputs, infer_outputs, Vec::<Vec<u8>>::new()).unwrap();

    let output_class = system_mem_zone_output.get_data(size_of_classes_output, 0);
    let reg_class = system_mem_zone_output.get_data(size_of_reg_output, size_of_classes_output);

    println!("{:?}", output_class);
    println!("{:?}", reg_class);

    println!("{:?}", unsafe { output_class.align_to::<f32>().1 });
    println!("{:?}", unsafe { reg_class.align_to::<f32>().1 });

    /* Unregister shared memory zone */
    let response = triton_inferer.unregister_system_shared_memory("image_input_data").unwrap();
    println!("{:?}", response);
    let response = triton_inferer.unregister_system_shared_memory("output_data").unwrap();
    println!("{:?}", response);

    system_mem_zone_input.destroy();
    system_mem_zone_output.destroy();

    Ok(())
}
