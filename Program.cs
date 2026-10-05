using OpenCvSharp;
using OpenCvSharp.Face;
using OpenCvSharp.Dnn;
using System.Security.Cryptography;

const string YunetUrl =
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx";

const string LbfUrl =
    "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml";

const string YunetHash =
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4";

// Папка без кириллицы.
string modelDir = @"C:\Users\Public\Documents\FacePointsModels";

Directory.CreateDirectory(modelDir);

string yunetPath = Path.Combine(
    modelDir,
    "face_detection_yunet_2023mar.onnx");

string lbfPath = Path.Combine(
    modelDir,
    "lbfmodel.yaml");

await DownloadYunetAsync(
    YunetUrl,
    yunetPath,
    YunetHash);

await DownloadIfMissingAsync(
    LbfUrl,
    lbfPath);

Console.WriteLine("Все модели готовы.");

using var camera =
    new VideoCapture(0, VideoCaptureAPIs.DSHOW);

if (!camera.IsOpened())
{
    Console.WriteLine("Не удалось открыть камеру.");
    return;
}

camera.Set(
    VideoCaptureProperties.FrameWidth,
    1280);

camera.Set(
    VideoCaptureProperties.FrameHeight,
    720);

using var frame = new Mat();
using var gray = new Mat();
using var display = new Mat();
using var faces = new Mat();

if (!camera.Read(frame) || frame.Empty())
{
    Console.WriteLine("Не удалось получить кадр.");
    return;
}

Console.WriteLine(
    $"Камера: {frame.Width}x{frame.Height}");

// ==========================================
// YuNet
// ==========================================

using var detector =
    FaceDetectorYN.Create(
        yunetPath,
        "",
        new Size(frame.Width, frame.Height),
        0.80f,
        0.30f,
        5000,
        Backend.DEFAULT,
        Target.CPU);

// ==========================================
// 68 точек лица
// ==========================================

using var facemark =
    FacemarkLBF.Create();

facemark.LoadModel(lbfPath);

// ==========================================
// Окно
// ==========================================

using var window =
    new Window(
        "FACE 68 POINTS - ESC для выхода",
        WindowFlags.AutoSize);

// ==========================================
// Главный цикл
// ==========================================

while (true)
{
    if (!camera.Read(frame) || frame.Empty())
        break;

    // --------------------------------------
    // Детекция лица
    // --------------------------------------

    detector.Detect(
        frame,
        faces);

    // --------------------------------------
    // Чёрно-белое изображение
    // --------------------------------------

    Cv2.CvtColor(
        frame,
        gray,
        ColorConversionCodes.BGR2GRAY);

    Cv2.EqualizeHist(
        gray,
        gray);

    Cv2.CvtColor(
        gray,
        display,
        ColorConversionCodes.GRAY2BGR);

    // --------------------------------------
    // Количество найденных лиц
    // --------------------------------------

    int faceCount = faces.Rows;

    if (faceCount == 0)
    {
        Cv2.PutText(
            display,
            "FACE NOT FOUND",
            new Point(20, 70),
            HersheyFonts.HersheySimplex,
            0.8,
            Scalar.White,
            2,
            LineTypes.AntiAlias);
    }

    // --------------------------------------
    // Собираем Rect[] для Facemark
    // --------------------------------------

    Rect[] detectedFaces =
        new Rect[faceCount];

    for (int r = 0; r < faceCount; r++)
    {
        float x =
            faces.At<float>(r, 0);

        float y =
            faces.At<float>(r, 1);

        float w =
            faces.At<float>(r, 2);

        float h =
            faces.At<float>(r, 3);

        int boxX =
            Math.Clamp(
                (int)x,
                0,
                frame.Width - 1);

        int boxY =
            Math.Clamp(
                (int)y,
                0,
                frame.Height - 1);

        int boxW =
            Math.Clamp(
                (int)w,
                1,
                frame.Width - boxX);

        int boxH =
            Math.Clamp(
                (int)h,
                1,
                frame.Height - boxY);

        detectedFaces[r] =
            new Rect(
                boxX,
                boxY,
                boxW,
                boxH);
    }

    // ======================================
    // Facemark
    // ======================================

    if (detectedFaces.Length > 0)
    {
        using InputArray faceInput =
            InputArray.Create(detectedFaces);

        Point2f[][] landmarks;

        bool success =
            facemark.Fit(
                gray,
                faceInput,
                out landmarks);

        if (success)
        {
            int landmarkFaceCount =
                Math.Min(
                    landmarks.Length,
                    detectedFaces.Length);

            for (int f = 0;
                     f < landmarkFaceCount;
                     f++)
            {
                Point2f[] points =
                    landmarks[f];

                Rect faceRect =
                    detectedFaces[f];

                // ----------------------------------
                // Рамка лица
                // ----------------------------------

                Cv2.Rectangle(
                    display,
                    faceRect,
                    Scalar.LimeGreen,
                    2,
                    LineTypes.AntiAlias);

                // ----------------------------------
                // 68 точек
                // ----------------------------------

                int pointCount =
                    points.Length;

                for (int i = 0;
                     i < pointCount;
                     i++)
                {
                    int px =
                        Math.Clamp(
                            (int)points[i].X,
                            0,
                            display.Width - 1);

                    int py =
                        Math.Clamp(
                            (int)points[i].Y,
                            0,
                            display.Height - 1);

                    var p =
                        new Point(px, py);

                    // Красная точка.
                    Cv2.Circle(
                        display,
                        p,
                        3,
                        Scalar.Red,
                        -1,
                        LineTypes.AntiAlias);

                    // Номер точки.
                    Cv2.PutText(
                        display,
                        i.ToString(),
                        new Point(
                            px + 5,
                            py - 5),
                        HersheyFonts.HersheySimplex,
                        0.32,
                        Scalar.White,
                        1,
                        LineTypes.AntiAlias);
                }

                // ----------------------------------
                // Контуры
                // ----------------------------------

                // Челюсть 0-16
                DrawOpenContour(
                    display,
                    points,
                    0,
                    16);

                // Левая бровь 17-21
                DrawOpenContour(
                    display,
                    points,
                    17,
                    21);

                // Правая бровь 22-26
                DrawOpenContour(
                    display,
                    points,
                    22,
                    26);

                // Нос 27-35
                DrawOpenContour(
                    display,
                    points,
                    27,
                    35);

                // Левый глаз 36-41
                DrawClosedContour(
                    display,
                    points,
                    36,
                    41);

                // Правый глаз 42-47
                DrawClosedContour(
                    display,
                    points,
                    42,
                    47);

                // Внешний рот 48-59
                DrawClosedContour(
                    display,
                    points,
                    48,
                    59);

                // Внутренний рот 60-67
                DrawClosedContour(
                    display,
                    points,
                    60,
                    67);

                // ----------------------------------
                // Информация
                // ----------------------------------

                float score =
                    faces.At<float>(f, 14);

                Cv2.PutText(
                    display,
                    $"68 POINTS | {score:0.00}",
                    new Point(
                        faceRect.X,
                        Math.Max(
                            25,
                            faceRect.Y - 10)),
                    HersheyFonts.HersheySimplex,
                    0.65,
                    Scalar.White,
                    2,
                    LineTypes.AntiAlias);
            }
        }
    }

    // ======================================
    // Верхняя надпись
    // ======================================

    Cv2.PutText(
        display,
        "BLACK & WHITE | 68 FACE POINTS | ESC - EXIT",
        new Point(20, 35),
        HersheyFonts.HersheySimplex,
        0.7,
        Scalar.White,
        2,
        LineTypes.AntiAlias);

    window.ShowImage(display);

    int key =
        Cv2.WaitKey(1);

    if (key == 27)
        break;
}

// ==========================================
// Открытый контур
// ==========================================

static void DrawOpenContour(
    Mat image,
    Point2f[] points,
    int start,
    int end)
{
    for (int i = start; i < end; i++)
    {
        var p1 =
            new Point(
                (int)points[i].X,
                (int)points[i].Y);

        var p2 =
            new Point(
                (int)points[i + 1].X,
                (int)points[i + 1].Y);

        Cv2.Line(
            image,
            p1,
            p2,
            Scalar.Yellow,
            1,
            LineTypes.AntiAlias);
    }
}

// ==========================================
// Замкнутый контур
// ==========================================

static void DrawClosedContour(
    Mat image,
    Point2f[] points,
    int start,
    int end)
{
    DrawOpenContour(
        image,
        points,
        start,
        end);

    var first =
        new Point(
            (int)points[start].X,
            (int)points[start].Y);

    var last =
        new Point(
            (int)points[end].X,
            (int)points[end].Y);

    Cv2.Line(
        image,
        last,
        first,
        Scalar.Yellow,
        1,
        LineTypes.AntiAlias);
}

// ==========================================
// Загрузка YuNet
// ==========================================

static async Task DownloadYunetAsync(
    string url,
    string path,
    string expectedHash)
{
    Console.WriteLine(
        "Проверка YuNet...");

    if (File.Exists(path))
    {
        string hash =
            await CalculateSha256Async(path);

        if (hash.Equals(
            expectedHash,
            StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine(
                "YuNet уже загружен.");

            return;
        }

        Console.WriteLine(
            "Файл YuNet неправильный. Удаляю.");

        File.Delete(path);
    }

    Console.WriteLine(
        "Загрузка YuNet...");

    using var client =
        new HttpClient();

    client.Timeout =
        TimeSpan.FromMinutes(5);

    using var response =
        await client.GetAsync(
            url,
            HttpCompletionOption.ResponseHeadersRead);

    response.EnsureSuccessStatusCode();

    await using Stream input =
        await response.Content.ReadAsStreamAsync();

    await using FileStream output =
        new FileStream(
            path,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None);

    await input.CopyToAsync(output);

    await output.FlushAsync();

    output.Close();

    string resultHash =
        await CalculateSha256Async(path);

    Console.WriteLine(
        $"SHA256: {resultHash}");

    if (!resultHash.Equals(
        expectedHash,
        StringComparison.OrdinalIgnoreCase))
    {
        File.Delete(path);

        throw new Exception(
            "YuNet скачан неправильно.");
    }

    Console.WriteLine(
        "YuNet успешно загружен.");
}

// ==========================================
// Обычная загрузка
// ==========================================

static async Task DownloadIfMissingAsync(
    string url,
    string path)
{
    if (File.Exists(path))
    {
        Console.WriteLine(
            $"{Path.GetFileName(path)} уже загружен.");

        return;
    }

    Console.WriteLine(
        $"Загрузка {Path.GetFileName(path)}...");

    using var client =
        new HttpClient();

    client.Timeout =
        TimeSpan.FromMinutes(5);

    using var response =
        await client.GetAsync(
            url,
            HttpCompletionOption.ResponseHeadersRead);

    response.EnsureSuccessStatusCode();

    await using Stream input =
        await response.Content.ReadAsStreamAsync();

    await using FileStream output =
        new FileStream(
            path,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None);

    await input.CopyToAsync(output);

    await output.FlushAsync();

    output.Close();

    Console.WriteLine(
        $"{Path.GetFileName(path)} готов.");
}

// ==========================================
// SHA256
// ==========================================

static async Task<string> CalculateSha256Async(
    string path)
{
    await using FileStream stream =
        File.OpenRead(path);

    using SHA256 sha256 =
        SHA256.Create();

    byte[] hash =
        await sha256.ComputeHashAsync(stream);

    return Convert.ToHexString(hash)
        .ToLowerInvariant();
}