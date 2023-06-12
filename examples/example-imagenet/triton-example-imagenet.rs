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

use std::process;
use std::error::Error;
use std::vec::Vec;
use ndarray::{array};
use nshare::ToNdarray3;

use triton_rust::TritonInference;
use triton_rust::inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};

fn main() -> Result<(), Box<dyn Error>> {
    let mut triton_inferer = TritonInference::connect("http://127.0.0.1:71").unwrap();

    /* Check if the model is running */
    let response = triton_inferer.is_model_ready("resnet18-imagenet", "1").unwrap();
    if response == false {
        println!("{}", "Model is not running");
        process::exit(1);
    }

    let _model_metadata = triton_inferer.get_model_metadata("resnet18-imagenet", "1").unwrap();

    let img = image::open("examples/example-imagenet/dog.jpeg").unwrap().into_rgb8();
    let img_ndarray = img.into_ndarray3();
    let mut img_ndarray_f32 = img_ndarray.mapv(|elem| (elem as f32) / 256.0);

    /* Create the array with mean and std for ImageNet */
    let imagenet_mean = array![0.485, 0.456, 0.406].mapv(|elem: f64| (elem as f32));
    let imagenet_std =  array![0.229, 0.224, 0.225].mapv(|elem: f64| (elem as f32));

    let mut imagenet_mean_broadcasted = imagenet_mean.broadcast((256, 256, 3)).unwrap();
    let mut imagenet_std_broadcasted = imagenet_std.broadcast((256, 256, 3)).unwrap();
    imagenet_mean_broadcasted.swap_axes(0,2);
    imagenet_std_broadcasted.swap_axes(0,2);

    /* Normalize the array */
    img_ndarray_f32 = (img_ndarray_f32 - imagenet_mean_broadcasted) / imagenet_std_broadcasted;

    /* Make the memory zone contiguous in standard layout */
    let img_ndarray_f32_continuous = img_ndarray_f32.as_standard_layout();

    let size_of_image: u64 = 3 * 256 * 256 * 4;
    let output_size: u64 = 1000 * 4;

    /* Create shared memory zones */
    let mut system_mem_zone_input = triton_inferer.create_system_shared_memory("input_data", "/input_data", size_of_image).unwrap();
    let mut system_mem_zone_output = triton_inferer.create_system_shared_memory("output_data", "/output_data", output_size).unwrap();

    /* Create input parameters */
    let mut infer_inputs = Vec::<InferInputTensor>::with_capacity(1);

    let input_params = triton_inferer.get_system_shared_memory_params("input_data", size_of_image, 0);
    infer_inputs.push(triton_inferer.get_infer_input("input_data", "FP32", &[3, 256, 256], input_params));

    system_mem_zone_input.copy_array(&img_ndarray_f32_continuous, 0);

    /* Create output parameters */
    let mut infer_outputs = Vec::<InferRequestedOutputTensor>::with_capacity(1);

    let outputs_params = triton_inferer.get_system_shared_memory_params("output_data", output_size, 0);
    infer_outputs.push(triton_inferer.get_infer_output("output_data", outputs_params));

    /* Inference */
    let _response  = triton_inferer.infer("resnet18-imagenet", "1", "25", infer_inputs, infer_outputs, Vec::<Vec<u8>>::new()).unwrap();

    /* Get the output */
    let outputs: Vec<f32> = system_mem_zone_output.get_data(output_size, 0);
    println!("{:?}", outputs);

    Ok(())
}
