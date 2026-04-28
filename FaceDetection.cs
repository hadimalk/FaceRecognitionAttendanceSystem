using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Drawing;

namespace FaceRecognitionAttendanceSystem {
    public class FaceDetection {
        private CascadeClassifier _cascadeClassifier;
        private Mat _frame;

        public FaceDetection(string cascadePath = "haarcascade_frontalface_default.xml") {
            _cascadeClassifier = new CascadeClassifier(cascadePath);
        }

        /// <summary>
        /// Detects faces in the provided image
        /// </summary>
        public Rectangle[] DetectFaces(Mat image) {
            if (image.IsEmpty) return new Rectangle[0];
            Mat gray = new Mat();
            CvInvoke.CvtColor(image, gray, ColorConversion.Bgr2Gray);
            CvInvoke.EqualizeHist(gray, gray);
            Rectangle[] faces = _cascadeClassifier.DetectMultiScale(
                gray, 1.1, 10, Size.Empty);
            return faces;
        }

        /// <summary>
        /// Draws rectangles around detected faces
        /// </summary>
        public void DrawFaces(Mat image, Rectangle[] faces) {
            foreach (Rectangle face in faces) {
                CvInvoke.Rectangle(image, face, new MCvScalar(0, 255, 0), 2);
            }
        }

        /// <summary>
        /// Extracts face region from image
        /// </summary>
        public Mat ExtractFace(Mat image, Rectangle faceRegion) {
            return new Mat(image, faceRegion);
        }
    }
}