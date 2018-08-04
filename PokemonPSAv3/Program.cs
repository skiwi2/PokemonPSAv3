using System;
using System.Collections.Generic;
using System.Configuration;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using OpenCvSharp;
using Tesseract;
using ZXing;

using Npgsql;

namespace PokemonPSAv3
{
    class Program
    {
        static void Main(string[] args)
        {
            var host = ConfigurationManager.AppSettings["host"];
            var username = ConfigurationManager.AppSettings["username"];
            var password = ConfigurationManager.AppSettings["password"];
            var database = ConfigurationManager.AppSettings["database"];

            var connectionString = $"Host={host};Username={username};Password={password};Database={database}";
            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();

                using (var command = new NpgsqlCommand("SELECT name FROM sets", connection))
                {
                    using (var reader = command.ExecuteReader())
                    {
                        while (reader.Read())
                        {
                            Console.WriteLine(reader.GetString(0));
                        }
                    }
                }
            }

            Console.Read();

            var barcodeReader = CreateBarcodeReader(BarcodeFormat.ITF, autoRotate: true, tryInverted: true, tryHarder: true);
            string barcode = null;

            var tesseractEngine = new TesseractEngine("./tessdata", "eng", EngineMode.Default);

            var imagePath = args[0];
            var sourceImage = new Mat(imagePath, ImreadModes.Color);
            ShowImage(sourceImage, "source");

            var thresholdedImage = GetRedThresholdedImage(sourceImage);
            ShowImage(thresholdedImage, "thresholded");

            for (int erodeIterations = 0; erodeIterations <= 4 && barcode == null; erodeIterations++)
            {
                var contours = GetCountours(thresholdedImage, erodeIterations);
                ShowContours(sourceImage, contours, new Scalar(0d, 255d, 0d), "contours " + erodeIterations);
                foreach (var contour in contours)
                {
                    if (barcode != null)
                    {
                        break;
                    }
                    using (var labelImage = GetCorrectedRectangle(sourceImage, contour))
                    {
                        var labelSize = labelImage.Size();
                        //if (contours.Length > 50 && labelSize.Height > 10 && labelSize.Width > 10)
                        if (labelSize.Height >= 50 && labelSize.Width >= labelSize.Height * 2.5)
                        {
                            //resize
                            while (labelImage.Size().Height < 250)
                            {
                                Cv2.Resize(labelImage, labelImage, new OpenCvSharp.Size(0d, 0d), 2d, 2d, InterpolationFlags.Lanczos4);
                            }

                            //sharpen
                            var tempMat = new Mat();
                            Cv2.GaussianBlur(labelImage, tempMat, new OpenCvSharp.Size(0d, 0d), 3d);
                            Cv2.AddWeighted(labelImage, 1.5d, tempMat, -0.5d, 0d, labelImage);

                            ShowImage(labelImage, "label " + contour.Length + " " + new Random().Next());

                            OCRText(tesseractEngine, labelImage);

                            using (var barcodeImage = GetBarcodeImage(labelImage))
                            {
                                //ShowImage(barcodeImage, "barcode " + new Random().Next());
                                barcode = barcode ?? DecodeBarcode(barcodeReader, barcodeImage);
                            }
                        }
                    }
                }
            }

            Console.WriteLine("Barcode: " + barcode);

            Cv2.WaitKey();
        }

        static Mat GetRedThresholdedImage(Mat sourceImage)
        {
            double minRedSaturation = 100d;
            double minRedValue = 100d;

            double maxRedSaturation = 255d;
            double maxRedValue = 255d;

            using (var hsvImage = new Mat())
            {
                Cv2.CvtColor(sourceImage, hsvImage, ColorConversionCodes.BGR2HSV);

                // first part of red is from 0 to 10 hue
                using (var lowerRed = new Mat())
                {
                    Cv2.InRange(hsvImage, new Scalar(0d, minRedSaturation, minRedValue), new Scalar(10d, maxRedSaturation, maxRedValue), lowerRed);

                    // second part of red is from 160 to 180 hue
                    using (var upperRed = new Mat())
                    {
                        Cv2.InRange(hsvImage, new Scalar(160d, minRedSaturation, minRedValue), new Scalar(180d, maxRedSaturation, maxRedValue), upperRed);

                        // add both parts together
                        var combinedRed = new Mat();
                        Cv2.AddWeighted(lowerRed, 1d, upperRed, 1d, 0d, combinedRed);

                        return combinedRed;
                    }
                }
            }
        }

        static OpenCvSharp.Point[][] GetCountours(Mat thresholdedImage, int erodeIterations)
        {
            using (var erodedImage = new Mat())
            {
                Cv2.Erode(thresholdedImage, erodedImage, new Mat(), iterations: erodeIterations);

                Cv2.FindContours(erodedImage, out var contours, out var hierarchyIndices, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

                return contours;
            }
        }

        static Mat GetCorrectedRectangle(Mat sourceImage, OpenCvSharp.Point[] contour)
        {
            var rotatedRectangle = Cv2.MinAreaRect(contour);

            var rectangleAngle = rotatedRectangle.Angle;
            var rectangleSize = ToSize(rotatedRectangle.Size);

            if (rectangleAngle < -45f)
            {
                rectangleAngle += 90f;
                rectangleSize = new OpenCvSharp.Size(rectangleSize.Height, rectangleSize.Width);
            }

            using (var rotationMatrix = Cv2.GetRotationMatrix2D(rotatedRectangle.Center, rectangleAngle, 1d))
            {
                using (var rotatedImage = new Mat())
                {
                    Cv2.WarpAffine(sourceImage, rotatedImage, rotationMatrix, sourceImage.Size(), InterpolationFlags.Lanczos4);

                    var croppedImage = new Mat();
                    Cv2.GetRectSubPix(rotatedImage, rectangleSize, rotatedRectangle.Center, croppedImage);

                    return croppedImage;
                }
            }
        }

        static OpenCvSharp.Size ToSize(Size2f size)
        {
            return new OpenCvSharp.Size(size.Width, size.Height);
        }

        static void ShowImage(Mat image, string imageName = "image")
        {
            using (new Window(imageName, image))
            {

            }
        }

        static void ShowContours(Mat sourceImage, OpenCvSharp.Point[][] contours, Scalar color, string imageName = "contours")
        {
            using (var contoursImage = new Mat())
            {
                sourceImage.CopyTo(contoursImage);

                Cv2.DrawContours(contoursImage, contours, -1, color);

                using (new Window(imageName, contoursImage))
                {

                }
            }
        }

        static Mat GetBarcodeImage(Mat labelImage)
        {
            // includes red border
            var labelRect = labelImage.Size();

            var width = labelRect.Width / 69d * 19d;    // 19mm width out of total 69mm label width
            var height = labelRect.Height / 20d * 4d;   // 4mm height out of total 20mm label height
            //var upperLeftX = labelRect.Width / 69d * 2d;  // 2mm from left on x-axis
            //var upperLeftY = labelRect.Height / 20d * 13d; // 13mm from top on y-axis

            var size = new OpenCvSharp.Size(width, height);
            var centerX = labelRect.Width / 69d * (2d + (19d / 2));     // 2mm from left on x-axis
            var centerY = labelRect.Height / 20d * (13d + (4d / 2));    // 13mm from top on y-axis

            return labelImage.GetRectSubPix(size, new Point2f((float)centerX, (float)centerY));
        }

        static BarcodeReader CreateBarcodeReader(BarcodeFormat barcodeFormat, bool autoRotate, bool tryInverted, bool tryHarder)
        {
            return new BarcodeReader
            {
                AutoRotate = autoRotate,
                TryInverted = tryInverted,
                Options =
                {
                    TryHarder = tryHarder,
                    PossibleFormats = new List<BarcodeFormat>
                    {
                        barcodeFormat
                    }
                }
            };
        }

        static string DecodeBarcode(BarcodeReader barcodeReader, Mat barcodeImage)
        {
            if (barcodeImage.Width == 0 || barcodeImage.Height == 0)
            {
                return null;
            }

            var barcodeBitmap = new Bitmap(barcodeImage.ToMemoryStream());

            var barcodeResult = barcodeReader.Decode(barcodeBitmap);
            if (barcodeResult != null && barcodeResult.Text.Length == 8)
            {
                return barcodeResult.Text;
            }
            return null;
        }

        static void OCRText(TesseractEngine tesseractEngine, Mat labelImage)
        {
            using (var page = tesseractEngine.Process(new Bitmap(labelImage.ToMemoryStream())))
            {
                Console.WriteLine(page.GetText());

                var wordDistance = GetWordDistance(page);
                Console.WriteLine("Word distance: " + wordDistance);
                if (wordDistance > -1)
                {
                    var lines = GetLines(page, wordDistance);

                    string game = "";
                    string numberInSet = "";
                    string cardName = "";
                    string grade = "";
                    string subset = "";
                    string serial = "";

                    if (lines.Count >= 1 && lines[0].Count >= 1)
                    {
                        game = string.Join(" ", lines[0].First().Words);
                    }

                    if (lines.Count >= 1 && lines[0].Count >= 2)
                    {
                        numberInSet = string.Join(" ", lines[0].Last().Words);
                    }

                    if (lines.Count >= 2 && lines[1].Count >= 1)
                    {
                        cardName = string.Join(" ", lines[1].First().Words);
                    }

                    if (lines.Count >= 2 && lines[1].Count >= 2)
                    {
                        grade = string.Join(" ", lines[1].Last().Words);
                    }

                    //subset is not always present, so only check if the words start on the first half of the label
                    if (lines.Count >= 3 && lines[2].Count >= 1 && lines[2][0].X1 <= page.RegionOfInterest.Width / 2d)
                    {
                        subset = string.Join(" ", lines[2].First().Words);
                    }

                    if (lines.Count >= 4 && lines[3].Count >= 2)
                    {
                        serial = string.Join(" ", lines[3].Last().Words);
                    }

                    Console.WriteLine($"Game: {game}");
                    Console.WriteLine($"Number in set: {numberInSet}");
                    Console.WriteLine($"Card name: {cardName}");
                    Console.WriteLine($"Grade: {grade}");
                    Console.WriteLine($"Subset: {subset}");
                    Console.WriteLine($"Serial: {serial}");
                }
            }
        }

        static int GetWordDistance(Page page)
        {
            using (var it = page.GetIterator())
            {
                it.Begin();

                do
                {
                    do
                    {
                        do
                        {
                            Tesseract.Rect? lastBoundingBox = null;

                            var textLine = it.GetText(PageIteratorLevel.TextLine);
                            if (textLine == null || textLine.Trim().Length == 0)
                            {
                                continue;
                            }
                            do
                            {
                                if (it.TryGetBoundingBox(PageIteratorLevel.Word, out var boundingBox))
                                {
                                    if (lastBoundingBox == null)
                                    {
                                        lastBoundingBox = boundingBox;
                                    }
                                    else if (lastBoundingBox != null)
                                    {
                                        var distance = boundingBox.X1 - lastBoundingBox.Value.X2;
                                        return distance;
                                    }
                                }
                                var text = it.GetText(PageIteratorLevel.Word);
                            }
                            while (it.Next(PageIteratorLevel.TextLine, PageIteratorLevel.Word));
                        }
                        while (it.Next(PageIteratorLevel.Para, PageIteratorLevel.TextLine));
                    }
                    while (it.Next(PageIteratorLevel.Block, PageIteratorLevel.Para));
                }
                while (it.Next(PageIteratorLevel.Block));
            }

            return -1;
        }

        static List<List<WordGroup>> GetLines(Page page, int wordDistance)
        {
            List<List<WordGroup>> lines = new List<List<WordGroup>>();

            using (var it = page.GetIterator())
            {
                it.Begin();

                do
                {
                    do
                    {
                        do
                        {
                            List<WordGroup> wordGroups = new List<WordGroup>();
                            List<string> words = new List<string>();
                            int x1 = 0;

                            Tesseract.Rect? lastBoundingBox = null;

                            var textLine = it.GetText(PageIteratorLevel.TextLine);
                            if (textLine.Trim().Length == 0 || (lines.Count == 0 && textLine.Split(' ').Length <= 1))
                            {
                                continue;
                            }
                            do
                            {
                                if (it.TryGetBoundingBox(PageIteratorLevel.Word, out var boundingBox))
                                {
                                    var word = it.GetText(PageIteratorLevel.Word);
                                    if (words.Count == 0 || boundingBox.X1 - lastBoundingBox.Value.X2 < 2 * wordDistance)
                                    {
                                        if (lastBoundingBox == null)
                                        {
                                            x1 = boundingBox.X1;
                                        }
                                        words.Add(word);
                                    }
                                    else
                                    {
                                        wordGroups.Add(new WordGroup(words, x1, lastBoundingBox.Value.X2));
                                        words = new List<string> { word };
                                        x1 = boundingBox.X1;
                                    }
                                    lastBoundingBox = boundingBox;
                                }
                            }
                            while (it.Next(PageIteratorLevel.TextLine, PageIteratorLevel.Word));

                            if (words.Count > 0)
                            {
                                wordGroups.Add(new WordGroup(words, x1, lastBoundingBox.Value.X2));
                            }
                            lines.Add(wordGroups);
                        }
                        while (it.Next(PageIteratorLevel.Para, PageIteratorLevel.TextLine));
                    }
                    while (it.Next(PageIteratorLevel.Block, PageIteratorLevel.Para));
                }
                while (it.Next(PageIteratorLevel.Block));
            }

            return lines;
        }
    }
}
