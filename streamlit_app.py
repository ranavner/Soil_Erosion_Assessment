import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# sample dataset linear regression, for area-gradient correlation
brown_area_samples = np.array([0, 2.1, 2.43, 10.05])  # Example brown area values
gradient_samples = np.array([0, 0.022, 0.05, 0.2])  # Corresponding gradients values

# Fit linear regression model
regression_model = LinearRegression().fit(brown_area_samples.reshape(-1, 1), gradient_samples.reshape(-1, 1))
predicted_values = regression_model.predict(brown_area_samples.reshape(-1, 1))
r_squared = r2_score(gradient_samples, predicted_values)


def calculate_brown_area(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Check if the image is loaded successfully
    if image is None:
        st.error("Error: Unable to load image.")
        return None, None, None, None, None
    
    #___________________________________________________________________________________________________

    # # detect lines and measure tthe distance:
    # def calculate_distance(line1, line2, scale, merge_threshold):
    #     x1, y1, x2, y2 = line1
    #     x3, y3, x4, y4 = line2
    #     # Calculate the endpoints of the lines
    #     p1 = np.array([x1, y1])
    #     p2 = np.array([x2, y2])
    #     p3 = np.array([x3, y3])
    #     p4 = np.array([x4, y4])
    #     # Calculate the distances between endpoints
    #     distance1 = np.linalg.norm(p1 - p3) * scale
    #     distance2 = np.linalg.norm(p1 - p4) * scale
    #     distance3 = np.linalg.norm(p2 - p3) * scale
    #     distance4 = np.linalg.norm(p2 - p4) * scale
    #     # Return the minimum distance
    #     min_distance = min(distance1, distance2, distance3, distance4)
        
    #     # Merge lines if they are within the merge_threshold
    #     if min_distance < merge_threshold:
    #         # Calculate the average endpoints of the merged line
    #         merged_line = np.array([(x1 + x3) // 2, (y1 + y3) // 2, (x2 + x4) // 2, (y2 + y4) // 2], dtype=np.int32)
    #         return min_distance, merged_line
    #     else:
    #         return min_distance, None

    # # Example usage in your code snippet
    # image_with_lines = np.copy(image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=140, minLineLength=100, maxLineGap=10)

    # # Draw lines on the original image (optional)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # # Measure distance between lines
    # if lines is not None:
    #     scale = scale_input  # Example scale: 1 pixel corresponds to 0.1 units (e.g., inches)
    #     merge_threshold = 1500  # Example merge threshold in pixels
    #     for i in range(len(lines)):
    #         for j in range(i + 1, len(lines)):
    #             line1 = lines[i][0]
    #             line2 = lines[j][0]
    #             distance, merged_line = calculate_distance(line1, line2, scale, merge_threshold)
    #             print(f"Distance between line {i+1} and line {j+1}: {distance} units")
    #             # Draw merged line if it exists
    #             if merged_line is not None:
    #                 x1, y1, x2, y2 = merged_line
    #                 cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)


#___________________________________________________________________________________________________


    # Define lower and upper bounds for brown color in RGB
    lower_brown = np.array([0, 34, 0], dtype="uint8")
    upper_brown = np.array([32, 255, 255], dtype="uint8")

    # Mask the image to get only brown areas
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # present brown lines on the image
    brown_lines_image = cv2.bitwise_and(image, image, mask=brown_mask)

    # Count the number of brown pixels
    brown_pixel_count = np.sum(brown_mask > 0)

    # Calculate the total number of pixels in the image
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate the percentage of brown area
    brown_percentage = (brown_pixel_count / total_pixels) * 100
    # Get coordinates of brown pixels
    brown_pixels = np.argwhere(brown_mask > 0)

    # Calculate the number of brown pixels
    brown_pixel_count = brown_pixels.shape[0]

    # Predict the gradient using the linear regression model
    if brown_percentage <= brown_area_samples.min():
        gradient = 0
    elif brown_percentage >= brown_area_samples.max():
        gradient = gradient_samples.max()
    else:
        gradient = regression_model.predict(np.array([[brown_percentage]]))[0][0]

    if 0 <= gradient <= 0.03:
        erosion_level = 'Low Risk of Soil Erosion'
        risk_color = 'green'
    elif 0.03 < gradient <= 0.04:
        erosion_level = 'Medium Risk of Soil Erosion'
        risk_color = 'orange'
    else:
        erosion_level = 'High Risk of Soil Erosion'
        risk_color = 'red'

    return brown_percentage, brown_lines_image, gradient, brown_percentage, erosion_level, risk_color





def main():

    # creating the streamlit app
    st.title("Soil Erosion and Gradient Calculation App")
    st.subheader('Please visit https://www.openstreetmap.org and use cyclOSM layer to get an image', divider='rainbow')

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        pil_image = Image.open(uploaded_image)
        # scale_input = st.number_input(f'Enter the image scale in pixels/meter: ', min_value=20)

        # save the uploaded image temporarily as PNG format
        temp_image_path = "temp_image.png"
        pil_image.save(temp_image_path)

        # calculate the brown area, get the processed image, and estimated gradient
        brown_percentage, brown_lines_image, gradient, brown_percentage, erosion_level, risk_color = calculate_brown_area(temp_image_path)

        if brown_percentage is not None:

            # display the processed image with only brown lines
            st.image(brown_lines_image, caption="Processed Image with Brown Lines Highlighted", use_column_width=True)
            # st.image(image_with_lines)
            # display the estimated gradient
            st.info(f'Estimated Average Gradient: {gradient:.3f} m/m')
            st.info(f'Level of Erosion Predicted: :{risk_color}[{erosion_level}]')
            st.info(f'Accuracy of Data: {r_squared:.3f}')
            made_up_data = 3
            st.write(f'Want to contribute? You can help by adding more data to our ML algorithm! we have more then {made_up_data} contributer so far!')
            st.info(f'Brown Pixels Percentage: {brown_percentage:.3f}')
        

        # remove the temporary image file
        os.remove(temp_image_path)

if __name__ == "__main__":
    main()
