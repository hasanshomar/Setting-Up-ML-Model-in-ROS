# Script for the `image_subscriber.py` Node
---

   ```python
   import os
   import cv2
   import numpy as np
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image
   from cv_bridge import CvBridge
   import tensorflow as tf
   from tensorflow.keras.models import load_model
   import time

   class ImageSubscriber(Node):
       def __init__(self):
           super().__init__('image_subscriber')
           self.subscription = self.create_subscription(
               Image,
               'image_topic',
               self.listener_callback,
               10)
           self.subscription  
           self.bridge = CvBridge()

           # Load the pre-trained model
           model_path = '/home/User/Downloads/Files/SegmentationLabwithcrop.keras' #change based on where you saved your model
           custom_objects = {
               'dice_loss': self.dice_loss,
               'dice_coefficient': self.dice_coefficient,
               'iou': self.iou
           }
           self.model = load_model(model_path, custom_objects=custom_objects)

           # Directory to save output images
           self.output_dir = '/path/to/save/output/images'  # Update with your desired directory

       @tf.keras.utils.register_keras_serializable()
       def dice_loss(self, y_true, y_pred):
           smooth = 1.
           y_true_f = tf.keras.backend.flatten(y_true)
           y_pred_f = tf.keras.backend.flatten(y_pred)
           intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
           return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

       @tf.keras.utils.register_keras_serializable()
       def dice_coefficient(self, y_true, y_pred):
           smooth = 1.
           y_true_f = tf.keras.backend.flatten(y_true)
           y_pred_f = tf.keras.backend.flatten(y_pred)
           intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
           return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

       @tf.keras.utils.register_keras_serializable()
       def iou(self, y_true, y_pred):
           smooth = 1.
           y_true_f = tf.keras.backend.flatten(y_true)
           y_pred_f = tf.keras.backend.flatten(y_pred)
           intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
           union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
           return (intersection + smooth) / (union + smooth)

       def preprocess_image(self, cv_image):
           gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
           clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
           enhanced_image = clahe.apply(gray_image)
           blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
           edges = cv2.Canny(blurred_image, 50, 150)
           kernel = np.ones((3, 3), np.uint8)
           dilated_edges = cv2.dilate(edges, kernel, iterations=1)
           eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)

           normalized_image = cv_image / 255.0
           normalized_image = np.expand_dims(normalized_image, axis=0)

           return normalized_image

       def save_images(self, base_filename, images):
           # Create a subfolder for each image's results
           subfolder = os.path.join(self.output_dir, f"{base_filename}_results")
           os.makedirs(subfolder, exist_ok=True)

           for idx, image in enumerate(images):
               save_path = os.path.join(subfolder, f"{base_filename}_output_{idx + 1}.png")
               cv2.imwrite(save_path, image)
               self.get_logger().info(f"Saved {save_path}")

       def listener_callback(self, msg):
           self.get_logger().info('Receiving image')

           # Convert ROS Image message to OpenCV image
           cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

           # Preprocess the image
           preprocessed_image = self.preprocess_image(cv_image)

           # Perform model inference
           start_time = time.perf_counter()
           predicted_mask = self.model.predict(preprocessed_image)
           end_time = time.perf_counter()
           elapsed_time = end_time - start_time
           self.get_logger().info(f"Model prediction took {elapsed_time:.8f} seconds.")

           # Post-process the prediction
           predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

           # Generate a base filename from the original image name (if available)
           base_filename = msg.header.frame_id.split('/')[-1].split('.')[0]

           # Save the images to the respective subfolder
           self.save_images(base_filename, [
               cv_image,  # Original image
               predicted_mask.squeeze(),  # Mask itself
               self.get_colored_overlay(cv_image, predicted_mask),  # Overlayed image
               self.get_mask_with_measurements(predicted_mask, self.calculate_layer_heights(predicted_mask))  # Mask with measurements
           ])

       def calculate_layer_heights(self, predicted_mask, region_width=40, scaling_factor=2):
           measurement_points = []
           scaled_mask = cv2.resize(predicted_mask.squeeze(), (predicted_mask.shape[2] * scaling_factor, predicted_mask.shape[1] * scaling_factor), interpolation=cv2.INTER_LINEAR)
           scaled_mask = cv2.GaussianBlur(scaled_mask, (7, 7), 0)
           contours, _ = cv2.findContours(scaled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
           contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

           num_layers = len(contours) - 1
           self.get_logger().info(f"Identified {num_layers} layers.")

           for i in range(num_layers):
               top_contour = contours[i]
               bottom_contour = contours[i+1]

               top_points = [pt[0] for pt in top_contour]
               bottom_points = [pt[0] for pt in bottom_contour]

               for x in range(0, scaled_mask.shape[1], region_width * scaling_factor):
                   top_y = min([pt[1] for pt in top_points if x - 15 <= pt[0] < x + region_width * scaling_factor + 15], default=None)
                   bottom_y = max([pt[1] for pt in bottom_points if x - 15 <= pt[0] < x + region_width * scaling_factor + 15], default=None)

                   if top_y is not None and bottom_y is not None:
                       height = abs(bottom_y - top_y) / scaling_factor
                       midpoint_y = (top_y + bottom_y) // 2
                       measurement_points.append((x // scaling_factor + region_width // 2, top_y // scaling_factor, bottom_y // scaling_factor, height, midpoint_y // scaling_factor))
                   else:
                       if measurement_points:
                           prev_x, prev_top_y, prev_bottom_y, prev_height, prev_midpoint_y = measurement_points[-1]
                           measurement_points.append((x // scaling_factor + region_width // 2, prev_top_y, prev_bottom_y, prev_height, prev_midpoint_y))

           return measurement_points

       def get_colored_overlay(self, original_image, predicted_mask):
           colored_mask = cv2.cvtColor(predicted_mask.squeeze() * 255, cv2.COLOR_GRAY2RGB)
           overlay = cv2.addWeighted(original_image, 0.5, colored_mask, 1, 0)
           return overlay

       def get_mask_with_measurements(self, predicted_mask, measurement_points):
           mask_with_points = cv2.cvtColor(predicted_mask.squeeze() * 255, cv2.COLOR_GRAY2RGB)

           for (x, top_y, bottom_y, height, midpoint_y) in measurement_points:
               cv2.circle(mask_with_points, (x, top_y), 2, (255, 0, 0), -1)
               cv2.circle(mask_with_points, (x, bottom_y), 2, (255, 0, 0), -1)
               cv2.line(mask_with_points, (x, top_y), (x, bottom_y), (255, 255, 255), 1)
               cv2.putText(mask_with_points, f"{int(height)}", (x + 5,

 midpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

           return mask_with_points

   def main(args=None):
       rclpy.init(args=args)
       image_subscriber = ImageSubscriber()
       rclpy.spin(image_subscriber)
       image_subscriber.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

## **Script Breakdown and Explanation**

### **1. Importing Libraries:**
```python
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
```
- **`os`:** Provides functions for interacting with the operating system, such as creating directories and handling file paths.
- **`cv2`:** OpenCV library used for computer vision tasks, such as image processing.
- **`numpy as np`:** NumPy is a library for numerical computations in Python, used here for handling arrays and image data.
- **`rclpy`:** The ROS 2 Python client library, used to create ROS nodes and interact with the ROS ecosystem.
- **`rclpy.node.Node`:** A base class to create a ROS 2 node in Python.
- **`sensor_msgs.msg.Image`:** The ROS message type for images, used to publish and subscribe to image topics.
- **`CvBridge`:** A ROS package that provides an interface between ROS and OpenCV, allowing for conversion between ROS images and OpenCV images.
- **`tensorflow as tf`:** TensorFlow is an open-source machine learning library, used here to load and run a pre-trained model.
- **`load_model`:** A function from TensorFlow to load a pre-trained Keras model.
- **`time`:** Provides various time-related functions, used here to measure the duration of model inference.

### **2. The `ImageSubscriber` Class:**
```python
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'image_topic',
            self.listener_callback,
            10)
        self.subscription  
        self.bridge = CvBridge()
```
- **`ImageSubscriber(Node)`:** A custom class that inherits from `Node`, making it a ROS 2 node. This class is responsible for subscribing to an image topic, processing the images using a machine learning model, and saving the results.
- **`super().__init__('image_subscriber')`:** Calls the constructor of the base `Node` class, naming the node `image_subscriber`.
- **`self.create_subscription`:** Creates a subscription to the `image_topic`, listening for incoming images of type `sensor_msgs.msg.Image`. The callback function `listener_callback` is called whenever a new image is received.
- **`self.bridge = CvBridge()`:** Initializes a `CvBridge` object, which is used to convert between ROS image messages and OpenCV images.

### **3. Loading the Pre-Trained Model:**
```python
model_path = '/home/User/Downloads/Files/SegmentationLabwithcrop.keras'  # Update the path accordingly
custom_objects = {
    'dice_loss': self.dice_loss,
    'dice_coefficient': self.dice_coefficient,
    'iou': self.iou
}
self.model = load_model(model_path, custom_objects=custom_objects)
```
- **`model_path`:** Specifies the path to the pre-trained Keras model. This path needs to be updated based on where the model is saved on your system.
- **`custom_objects`:** A dictionary that maps the custom loss and metric functions (`dice_loss`, `dice_coefficient`, `iou`) to their implementations. This is necessary because the model was trained with these custom functions.
- **`load_model`:** Loads the pre-trained Keras model, using the custom objects specified.

### **4. Directory to Save Output Images:**
```python
self.output_dir = '/path/to/save/output/images'  # Update with your desired directory
```
- **`self.output_dir`:** Specifies the directory where the processed images and results will be saved. Update this path based on your directory structure.

### **5. Custom Loss and Metric Functions:**
```python
@tf.keras.utils.register_keras_serializable()
def dice_loss(self, y_true, y_pred):
    # Implementation of the Dice loss function

@tf.keras.utils.register_keras_serializable()
def dice_coefficient(self, y_true, y_pred):
    # Implementation of the Dice coefficient

@tf.keras.utils.register_keras_serializable()
def iou(self, y_true, y_pred):
    # Implementation of the Intersection over Union (IoU) metric
```
- **`@tf.keras.utils.register_keras_serializable()`**: Registers the custom functions with Keras, allowing them to be used with the loaded model.
- **`dice_loss`, `dice_coefficient`, `iou`:** These functions are used to evaluate the performance of the model during training and inference. They are common metrics in image segmentation tasks.

### **6. Preprocessing the Image:**
```python
def preprocess_image(self, cv_image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(gray_image)
    
    # Apply Gaussian blur and edge detection
    blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    
    # Perform dilation and erosion
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)
    
    # Normalize the image for model input
    normalized_image = cv_image / 255.0
    normalized_image = np.expand_dims(normalized_image, axis=0)
    
    return normalized_image
```
- **`preprocess_image`:** This function preprocesses the input image before it is passed to the machine learning model. It includes steps like converting to grayscale, contrast enhancement, blurring, edge detection, and normalization.

### **7. Saving the Processed Images:**
```python
def save_images(self, base_filename, images):
    # Create a subfolder for each image's results
    subfolder = os.path.join(self.output_dir, f"{base_filename}_results")
    os.makedirs(subfolder, exist_ok=True)

    for idx, image in enumerate(images):
        save_path = os.path.join(subfolder, f"{base_filename}_output_{idx + 1}.png")
        cv2.imwrite(save_path, image)
        self.get_logger().info(f"Saved {save_path}")
```
- **`save_images`:** Saves the processed images in a subfolder named after the original image file with `_results` appended. The images are saved with descriptive filenames like `output_1.png`, `output_2.png`, etc.

### **8. Callback Function for Image Subscription:**
```python
def listener_callback(self, msg):
    self.get_logger().info('Receiving image')
    
    # Convert ROS Image message to OpenCV image
    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    
    # Preprocess the image
    preprocessed_image = self.preprocess_image(cv_image)
    
    # Perform model inference
    start_time = time.perf_counter()
    predicted_mask = self.model.predict(preprocessed_image)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    self.get_logger().info(f"Model prediction took {elapsed_time:.8f} seconds.")
    
    # Post-process the prediction
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    
    # Generate a base filename from the original image name (if available)
    base_filename = msg.header.frame_id.split('/')[-1].split('.')[0]
    
    # Save the images to the respective subfolder
    self.save_images(base_filename, [
        cv_image,  # Original image
        predicted_mask.squeeze(),  # Mask itself
        self.get_colored_overlay(cv_image, predicted_mask),  # Overlayed image
        self.get_mask_with_measurements(predicted_mask, self.calculate_layer_heights(predicted_mask))  # Mask with measurements
    ])
```
- **`listener_callback`:** This is the callback function that gets triggered every time an image is received on the `image_topic`. It processes the image through the machine learning model, logs the time taken for inference, and then saves the results.
- **`base_filename`:** Extracts the original image's filename from the message header, ensuring the results are saved with a consistent naming convention.

### **9. Calculating Layer Heights:**
```python
def calculate_layer_heights(self, predicted_mask, region_width=40, scaling_factor=2):
    # Calculate the height of layers based on the contours of the predicted mask
    ...
```
- **`calculate_layer_heights`:** This function analyzes the predicted mask to identify and measure the height of individual layers in the image. It does this by finding contours and calculating the vertical distance between them.

### **10. Generating Overlay Images:**
```python
def get_colored_overlay(self, original_image, predicted_mask):
    # Generate an overlay of the mask on the original image
    ...

def get_mask_with_measurements(self, predicted_mask, measurement_points):
    # Add measurement points and lines to the mask
    ...
```
- **`get_colored_overlay`:** Creates a

 visualization by overlaying the predicted mask onto the original image.
- **`get_mask_with_measurements`:** Adds the calculated layer heights as annotations on the mask, showing the measurements directly on the image.

### **11. Main Function:**
```python
def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```
- **`main`:** Initializes the ROS 2 Python client, creates an instance of the `ImageSubscriber` node, and keeps it running to continuously process incoming images. When the node is shut down, it destroys the node and cleanly shuts down the ROS 2 system.
