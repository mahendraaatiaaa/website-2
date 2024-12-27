// URL model ONNX
const MODEL_URL = './model_training_1.onnx';

// DOM Elements
const imageUpload = document.getElementById('imageUpload');
const uploadedImage = document.getElementById('uploadedImage');
const resultsDiv = document.getElementById('results');

// Variables
let model;

// Load the ONNX model
async function loadModel() {
    try {
        model = await ort.InferenceSession.create(MODEL_URL);
        console.log('Model loaded successfully');
        console.log('Input Names:', model.inputNames);
        console.log('Output Names:', model.outputNames);
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

// Preprocess image to match model input
function preprocessImage(image) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Set canvas size and draw image
    canvas.width = 177;
    canvas.height = 177;
    ctx.drawImage(image, 0, 0, 177, 177);

    // Extract image data and normalize
    const imageData = ctx.getImageData(0, 0, 177, 177);
    const { data } = imageData;
    const input = new Float32Array(3 * 177 * 177);

    for (let i = 0; i < data.length; i += 4) {
        const pixelIndex = i / 4;
        const rowIndex = pixelIndex % 177;
        const colIndex = Math.floor(pixelIndex / 177);
        input[rowIndex + 177 * colIndex] = data[i] / 255.0; // R
        input[rowIndex + 177 * colIndex + 177 * 177] = data[i + 1] / 255.0; // G
        input[rowIndex + 177 * colIndex + 2 * 177 * 177] = data[i + 2] / 255.0; // B
    }

    return new ort.Tensor('float32', input, [1, 3, 177, 177]);
}

// Upload and classify image
async function uploadAndClassifyImage() {
    const file = imageUpload.files[0];
    if (!file) {
        resultsDiv.innerHTML = '<p>Please upload an image to classify.</p>';
        return;
    }

    const reader = new FileReader();

    // Display uploaded image
    reader.onloadend = async function () {
        uploadedImage.src = reader.result;

        const image = new Image();
        image.src = reader.result;

        image.onload = async () => {
            resultsDiv.innerHTML = '<p>Classifying... Please wait.</p>';

            try {
                const tensor = preprocessImage(image);

                // Validate model
                if (!model) {
                    throw new Error('Model is not loaded. Please refresh the page.');
                }

                // Use model input and output names
                const feeds = { [model.inputNames[0]]: tensor };
                const output = await model.run(feeds);

                const probabilities = output[model.outputNames[0]].data;
                const classNames = [
                    'anggur', 'apel', 'belimbing', 'jeruk', 'kiwi',
                    'mangga', 'nanas', 'pisang', 'semangka', 'stroberi'
                ];

                // Find class with the highest probability
                const maxIndex = probabilities.indexOf(Math.max(...probabilities));

                // Clear results div and display classification results
                resultsDiv.innerHTML = '';
                probabilities.forEach((prob, index) => {
                    resultsDiv.innerHTML += `
                        <div class="result">
                            <span>${classNames[index].toUpperCase()}</span>
                            <div class="bar-container">
                                <div class="bar" style="width: ${(prob * 100).toFixed(2)}%;"></div>
                            </div>
                            <span>${(prob * 100).toFixed(2)}%</span>
                        </div>
                    `;
                });

                // Output the highest probability and the class name
                resultsDiv.innerHTML += `
                    <h3>Most Likely Class:</h3>
                    <p>${classNames[maxIndex].toUpperCase()} with ${(probabilities[maxIndex] * 100).toFixed(2)}%</p>
                `;
            } catch (error) {
                console.error('Error classifying image:', error);
                resultsDiv.innerHTML = '<p>Error classifying image. Please check the console for details.</p>';
            }
        };
    };

    reader.readAsDataURL(file);
}

// Load model on page load
loadModel();

// Add event listener to classify image on file upload
imageUpload.addEventListener('change', uploadAndClassifyImage);
