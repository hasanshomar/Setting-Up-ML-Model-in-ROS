# Create ROS2 Workspace and packages
---
This section explains how to set up a new ros2 workspace for the project's operations. It will guide you through the commands to run and what each command does, how to create the workspace, and how to create the packages within the workspace. 

---
### **Step 1: Create the ROS 2 Workspace**

*For the sake of example we will proceed with creating a workspace called `ros2_ws_test`.*

1. **Open a Terminal**:
   - Start by opening a terminal on your Ubuntu system.

2. **Create the Workspace Directory**:
   - Navigate to the location where you want to create your workspace, then create the workspace directory named `ros2_ws_test`:

   ```bash
   mkdir -p ~/ros2_ws_test/src
   cd ~/ros2_ws_test
   ```

3. **Initialize the Workspace**:
   - Initialize the workspace by running the following command:

   ```bash
   colcon build
   ```

4. **Source the Workspace**:
   - Source the setup file to overlay the workspace on top of your current environment:

   ```bash
   source install/setup.bash
   ```

   To make this automatic for every new terminal session, add the source command to your `.bashrc` file:

   ```bash
   echo "source ~/ros2_ws_test/install/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

### **Step 2: Create a New ROS 2 Package**

1. **Navigate to the Source Directory**:
   - Move into the `src` directory of your workspace:

   ```bash
   cd ~/ros2_ws_test/src
   ```

2. **Create a New Package**:
   - Use the following command to create a new package. You can name it something like `image_processor_test`:

   ```bash
   ros2 pkg create --build-type ament_python image_processor_test
   ```

3. **Verify the Package Creation**:
   - Navigate into the package directory to see the structure:

   ```bash
   cd image_processor_test
   ```

### **Step 3: Build the Workspace**

1. **Return to the Workspace Root**:
   - Navigate back to the root of your workspace:

   ```bash
   cd ~/ros2_ws_test
   ```

2. **Build the Workspace**:
   - Run the `colcon build` command to build the workspace, including the new package:

   ```bash
   colcon build
   ```

3. **Source the Workspace**:
   - Source the workspace after building:

   ```bash
   source install/setup.bash
   ```
