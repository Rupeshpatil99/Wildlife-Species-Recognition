import { useState } from "react";
import axios from "axios";

function App() {

  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);


  const handleImageChange = (event) => {

    const selectedImage = event.target.files[0];

    if (!selectedImage) {
      return;
    }

    setImage(selectedImage);
    setPreview(URL.createObjectURL(selectedImage));
    setResult(null);
  };


  const handlePredict = async () => {

    if (!image) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();

    formData.append("image", image);

    try {

      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData
      );

      setResult(response.data);

    } catch (error) {

      console.error(error);

      alert("Prediction failed. Make sure Flask is running.");

    } finally {

      setLoading(false);
    }
  };


  return (

    <div>

      <h1>Wildlife Species Recognition</h1>

      <input
        type="file"
        accept="image/*"
        onChange={handleImageChange}
      />

      {preview && (
        <div>
          <img
            src={preview}
            alt="Preview"
            width="300"
          />
        </div>
      )}

      <button onClick={handlePredict}>
        {loading ? "Predicting..." : "Predict"}
      </button>


      {result && (

        <div>

          <h2>Prediction Result</h2>

          <h3>
            Species: {result.prediction}
          </h3>

          <p>
            Confidence: {result.confidence}%
          </p>

          <p>
            Status: {result.status}
          </p>


          <h3>Top 3 Predictions</h3>

          <ol>

            {result.top_3.map((item, index) => (

              <li key={index}>

                {item.species} — {item.confidence}%

              </li>

            ))}

          </ol>

        </div>

      )}

    </div>
  );
}

export default App;