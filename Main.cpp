// Animation2D.cpp : Defines the entry point for the application.
//
#define _CRTDBG_MAP_ALLOC
#include <Windows.h>

#include <stdlib.h>
#include <crtdbg.h>

#include <fstream>
#include <vector>

#include <d2d1.h>
#include <d2d1_1.h>
#include <dwrite.h>

#pragma comment(lib, "D2D1.lib")
#pragma comment(lib, "Dwrite.lib")
#pragma comment(lib, "Dxguid.lib")
#pragma comment(lib, "Windowscodecs.lib")


#include <wincodec.h>
#include <string>
#include <vector>
#include <ctime>

#include "NeuralNetwork.h"

#ifdef _DEBUG
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
// Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
// allocations to be of _CLIENT_BLOCK type
#define new DBG_NEW
#else
#define DBG_NEW new
#endif


#ifndef _UNICODE
    typedef CHAR Char;
#define Text(t) (t)
#else
    typedef WCHAR Char;
#define Text(t) L##t
#endif

using namespace std;
using namespace D2D1;

// Global Variables:
HINSTANCE hInst;                                // current instance
const Char *szTitle;                  // The title bar text
const Char *szWindowClass;            // the main window class name
HWND  g_hWnd;


ID2D1Factory1         *g_pFactory = NULL;
IDWriteFactory        *g_pWriteFactory = NULL;
IWICImagingFactory    *g_pWICImageingFactory = NULL;

IDWriteTextFormat     *g_pTextFormat = NULL;


int g_nWidth = 700;
int g_nHeight = 700;


// Forward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);

HRESULT             InitD2D();
void                CleanUpD2D();

HRESULT InitD2D()
{
    HRESULT hr;
    CoInitialize(NULL);
    hr = CoCreateInstance(
        CLSID_WICImagingFactory,
        NULL,
        CLSCTX_INPROC_SERVER,
        IID_IWICImagingFactory,
        (LPVOID*)&g_pWICImageingFactory
    );
    hr = D2D1CreateFactory(D2D1_FACTORY_TYPE::D2D1_FACTORY_TYPE_SINGLE_THREADED, &g_pFactory);
    if (FAILED(hr))
        return hr;

    hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE::DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), (IUnknown**)&g_pWriteFactory);
    hr = g_pWriteFactory->CreateTextFormat(L"Microsoft Sans Serif", NULL, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 20, L"", &g_pTextFormat);
    g_pTextFormat->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_CENTER);

    g_pTextFormat->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);


    return hr;
}
void CleanUpD2D()
{
    if (g_pFactory != NULL)
    {
        g_pFactory->Release();
        g_pFactory = NULL;
    }
    if (g_pTextFormat != NULL)
    {
        g_pTextFormat->Release();
        g_pTextFormat = NULL;
    }
    if (g_pWriteFactory != NULL)
    {
        g_pWriteFactory->Release();
        g_pWriteFactory = NULL;
    }
    if (g_pWICImageingFactory != NULL)
    {
        g_pWICImageingFactory->Release();
        g_pWICImageingFactory = NULL;
    }

    CoUninitialize();
}

void Refresh(HWND hWnd)
{
    RECT rect;
    GetClientRect(hWnd, &rect);
    InvalidateRect(hWnd, &rect, TRUE);
    UpdateWindow(hWnd);
}


// unsigned char *image;
NeuralNetwork net;

vector<vector<D2D1_POINT_2F>> *g_Strokes;
bool g_bIsDown = false;

bool g_bHasAnswer = false;
int g_nAnswer;
float g_fConfidency;

float g_DrawCanvasSize = 500;

unsigned char image[28 * 28];

void GetAnswer()
{
    g_bHasAnswer = true;

    IWICBitmap *pBitmap = NULL;
    HRESULT hr = g_pWICImageingFactory->CreateBitmap(28, 28, GUID_WICPixelFormat32bppRGB, WICBitmapCacheOnDemand, &pBitmap);

    ID2D1RenderTarget *pRenderTarget = NULL;
    hr = g_pFactory->CreateWicBitmapRenderTarget(pBitmap, D2D1::RenderTargetProperties(), &pRenderTarget);

    ID2D1SolidColorBrush *pBrush;

    pRenderTarget->BeginDraw();
    pRenderTarget->Clear(ColorF(ColorF::White));

    pRenderTarget->CreateSolidColorBrush(ColorF(ColorF::Black), &pBrush);


    float lettersize, offsetX, offsetY;
    {

        float leftMost = FLT_MAX, rightMost = -FLT_MAX, upMost = FLT_MAX, downMost = -FLT_MAX;

        for (int i = 0; i < g_Strokes->size(); i++)
        {
            for (int j = 0; j < g_Strokes->at(i).size(); j++)
            {
                D2D1_POINT_2F p = g_Strokes->at(i)[j];
                p.x = p.x;
                p.y = p.y;
                if (p.x < leftMost)
                    leftMost = p.x;
                if (p.x > rightMost)
                    rightMost = p.x;
                if (p.y < upMost)
                    upMost = p.y;
                if (p.y > downMost)
                    downMost = p.y;
            }
        }

        float sizeX = rightMost - leftMost;
        float sizeY = downMost - upMost;

        offsetX = leftMost + sizeX * 0.5f;
        offsetY = upMost + sizeY * 0.5f;
        lettersize = max(sizeX, sizeY);
    }

    float letterScale = 16 / 28.0f;
    float letteroffset = 6.0f / 28.0f;
    for (int i = 0; i < g_Strokes->size(); i++)
    {
        for (int j = 0; j < g_Strokes->at(i).size(); j++)
        {
            D2D1_POINT_2F p = g_Strokes->at(i)[j];
            p.x = ((p.x - offsetX) / lettersize + 0.5f) * letterScale + letteroffset;
            p.y = ((p.y - offsetY) / lettersize + 0.5f) * letterScale + letteroffset;

            g_Strokes->at(i)[j] = p;
        }
    }

    ID2D1StrokeStyle *pStyle = NULL;

    g_pFactory->CreateStrokeStyle(
        StrokeStyleProperties(D2D1_CAP_STYLE_ROUND,
            D2D1_CAP_STYLE_ROUND,
            D2D1_CAP_STYLE_ROUND,
            D2D1_LINE_JOIN_BEVEL), NULL, 0, &pStyle);

    for (int i = 0; i < g_Strokes->size(); i++)
    {
        if (g_Strokes->at(i).size() <= 1)
            continue;

        ID2D1PathGeometry *pGeometry;
        g_pFactory->CreatePathGeometry(&pGeometry);

        ID2D1GeometrySink *pSink;
        pGeometry->Open(&pSink);

        pSink->BeginFigure(Point2F(g_Strokes->at(i)[0].x * 28, g_Strokes->at(i)[0].y * 28), D2D1_FIGURE_BEGIN_HOLLOW);

        for (int j = 1; j < g_Strokes->at(i).size(); j++)
        {
            pSink->AddLine(Point2F(g_Strokes->at(i)[j].x * 28, g_Strokes->at(i)[j].y * 28));
        }
        pSink->EndFigure(D2D1_FIGURE_END_OPEN);
        pSink->Close();
        pSink->Release();

        pRenderTarget->DrawGeometry(pGeometry, pBrush, 1.7, pStyle);

        pGeometry->Release();
    }


    pStyle->Release();
    pBrush->Release();
    pRenderTarget->EndDraw();
    pRenderTarget->Release();

    IWICBitmapLock *pBitmapLock;

    WICRect rect;
    rect.X = 0;
    rect.Y = 0;
    rect.Width = 28;
    rect.Height = 28;
    pBitmap->Lock(&rect, WICBitmapLockRead, &pBitmapLock);

    UINT size;
    UINT *pixels;
    hr = pBitmapLock->GetDataPointer(&size, (BYTE**)&pixels);

    Input input;
    ForwardResult result;
    net.Initialize(&input);
    net.Initialize(&result);

    for (int i = 0; i < 28 * 28; i++)
    {
        input.m_Input(i, 0) = (255 - (pixels[i] & 255)) / 255.0f;
        image[i] = (255 - (pixels[i] & 255));
    }

    net.Forward(input, &result);
    Float max = -_FLOAT_MAX;
    int index = 0;
    for (int j = 0; j < 10; j++)
    {
        Float f = result.m_LayerAs[result.m_nCount - 1](j, 0);
        if (f > max)
        {
            max = f;
            index = j;
        }
        Float k = 0;
    }

    g_fConfidency = max;

    net.Delete(&input);
    net.Delete(&result);

    pBitmapLock->Release();

    pBitmap->Release();

    g_nAnswer = index;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_ERASEBKGND:
    {
        ID2D1HwndRenderTarget *pRenderTarget = NULL;
        HRESULT hr = g_pFactory->CreateHwndRenderTarget(D2D1::RenderTargetProperties(), D2D1::HwndRenderTargetProperties(g_hWnd, SizeU(g_nWidth, g_nHeight)), &pRenderTarget);

        pRenderTarget->BeginDraw();
        pRenderTarget->Clear(ColorF(ColorF::White));

        ID2D1SolidColorBrush *pBrush;

        pRenderTarget->CreateSolidColorBrush(ColorF(ColorF::Black), &pBrush);

        ID2D1StrokeStyle *pStyle = NULL;

        g_pFactory->CreateStrokeStyle(
            StrokeStyleProperties(D2D1_CAP_STYLE_ROUND,
                D2D1_CAP_STYLE_ROUND,
                D2D1_CAP_STYLE_ROUND,
                D2D1_LINE_JOIN_BEVEL), NULL, 0, &pStyle);

        if (!g_bHasAnswer)
        {
            for (int i = 0; i < g_Strokes->size(); i++)
            {
                if (g_Strokes->at(i).size() <= 1)
                    continue;

                ID2D1PathGeometry *pGeometry;
                g_pFactory->CreatePathGeometry(&pGeometry);

                ID2D1GeometrySink *pSink;
                pGeometry->Open(&pSink);

                pSink->BeginFigure(g_Strokes->at(i)[0], D2D1_FIGURE_BEGIN_HOLLOW);

                for (int j = 1; j < g_Strokes->at(i).size(); j++)
                {
                    pSink->AddLine(g_Strokes->at(i)[j]);
                }
                pSink->EndFigure(D2D1_FIGURE_END_OPEN);
                pSink->Close();
                pSink->Release();

                pRenderTarget->DrawGeometry(pGeometry, pBrush, 20, pStyle);

                pGeometry->Release();
            }
        }
        else
        {

            float pixelSize = g_DrawCanvasSize / 28.0f;

            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    unsigned char c1 = image[j * 28 + i];
                    float f = 1 - (c1) / (255.0f);
                    pBrush->SetColor(ColorF(f, f, f));
                    pRenderTarget->FillRectangle(RectF(pixelSize * i, pixelSize * j, pixelSize * (i + 1), pixelSize * (j + 1)), pBrush);
                }
            }
        }

        pStyle->Release();
        pBrush->SetColor(ColorF(0x00909090, 1.0f));

        pRenderTarget->FillRectangle(RectF(g_DrawCanvasSize, 0, 700, 700), pBrush);
        pRenderTarget->FillRectangle(RectF(0, g_DrawCanvasSize, g_DrawCanvasSize, 700), pBrush);

        pBrush->SetColor(ColorF(0x00F0F0F0, 1.0f));
        pRenderTarget->FillRectangle(RectF(150, 550, 350, 650), pBrush);

        pBrush->SetColor(ColorF(ColorF::Black));
        auto text = L"Convert";
        pRenderTarget->DrawTextA(text, 7, g_pTextFormat, RectF(150, 550, 350, 650), pBrush);

        if (g_bHasAnswer)
        {
            wchar_t num[1];
            num[0] = (wchar_t)(g_nAnswer + '0');
            pRenderTarget->DrawTextA(num, 1, g_pTextFormat, RectF(550, 50, 650, 150), pBrush);


            std::wstring s = to_wstring(g_fConfidency);

            pRenderTarget->DrawTextA(s.c_str(), s.size(), g_pTextFormat, RectF(550, 150, 650, 250), pBrush);
        }
        //pRenderTarget->DrawLine(Point2F(0, 0), Point2F(300, 300), pBrush);

        pBrush->Release();
        pRenderTarget->EndDraw();

        pRenderTarget->Release();

        break;
    }
    case WM_EXITSIZEMOVE:
    {
        RECT rect;
        GetClientRect(hWnd, &rect);

        g_nWidth = rect.right - rect.left;
        g_nHeight = rect.bottom - rect.top;

        break;
    }
    case WM_LBUTTONDOWN:
    {
        int xPos = (lParam & 0x0000FFFF);
        int yPos = (lParam >> 16);

        if (xPos < 0 || xPos > g_DrawCanvasSize || yPos < 0 || yPos > g_DrawCanvasSize)
            break;

        if (g_bHasAnswer)
        {
            g_Strokes->clear();
        }
        g_bHasAnswer = false;
        g_Strokes->push_back(vector<D2D1_POINT_2F>());
        g_bIsDown = true;

        Refresh(hWnd);
        break;
    }
    case WM_MOUSEMOVE:
    {
        if (g_bIsDown)
        {
            float xPos = (lParam & 0x0000FFFF);
            float yPos = (lParam >> 16);

            g_Strokes->at(g_Strokes->size() - 1).push_back(Point2F(xPos, yPos));
            Refresh(hWnd);
        }
        break;
    }
    case WM_LBUTTONUP:
    {
        int xPos = (lParam & 0x0000FFFF);
        int yPos = (lParam >> 16);

        if (!g_bIsDown)
        {
            if (xPos > 150 && xPos < 350 && yPos > 550 && yPos < 650)
            {
                GetAnswer();
                Refresh(hWnd);
                break;
            }
        }
        g_bIsDown = false;
        break;
    }
    case WM_KEYUP:
    {
        switch (wParam)
        {
        case VK_RETURN:
            GetAnswer();
            Refresh(hWnd);
            break;
        }
        break;
    }
    case WM_DESTROY:
    {
        PostQuitMessage(0);
        break;
    }
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEX wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPLICATION));
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = NULL;
    wcex.lpszClassName = szWindowClass;
    wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_APPLICATION));

    return RegisterClassEx(&wcex);
}
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
    hInst = hInstance; // Store instance handle in our global variable

    g_hWnd = CreateWindow(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, 0, g_nWidth + 16, g_nHeight + 39, NULL, NULL, hInstance, NULL);

    if (!g_hWnd)
    {
        return FALSE;
    }

    ShowWindow(g_hWnd, nCmdShow);
    UpdateWindow(g_hWnd);

    return TRUE;
}

struct Image
{
    unsigned char pixels[28][28];
    unsigned char num;
};
struct ImageSet
{
    Image *images;
    int count;
};

ImageSet ReadImageset(string sFileName)
{
    ImageSet set;

    ifstream fimage("TestData\\" + sFileName + "_image.dat", ios::in | ios::binary | ios::ate);
    ifstream flabel("TestData\\" + sFileName + "_label.dat", ios::in | ios::binary | ios::ate);

    fimage.seekg(0, ios::beg);
    flabel.seekg(0, ios::beg);

    if ((!fimage.is_open()) || (!flabel.is_open()))
        return {};

    int magicNumber = 0;
    unsigned char c[4];
    fimage.read((char*)c, 4);
    flabel.read((char*)c, 4);

    int count = 0;
    fimage.read((char*)c, 4);
    flabel.read((char*)c, 4);
    count = ((c[0] * 256 + c[1]) * 256 + c[2]) * 256 + c[3];

    set.count = count;
    set.images = new Image[count];

    int rowNum, columnNum;
    fimage.read((char*)c, 4);
    rowNum = ((c[0] * 256 + c[1]) * 256 + c[2]) * 256 + c[3];
    fimage.read((char*)c, 4);
    columnNum = ((c[0] * 256 + c[1]) * 256 + c[2]) * 256 + c[3];

    for (int i = 0; i < count; i++)
    {
        fimage.read((char*)(set.images[i].pixels), rowNum * columnNum);
        flabel.read((char*)(&set.images[i].num), 1);
    }

    fimage.close();
    flabel.close();

    return set;
}

void SaveProgress(std::string fileName, std::vector<Progression> *progression)
{
    ofstream fout(fileName, std::ios::out | std::ios::binary | std::ios::ate);


    int magicNum = 0x9fb8a6c3;
    fout.write((char*)(&magicNum), 4);

    int count = progression->size();
    fout.write((char*)(&count), 4);

    for (int i = 0; i < count; i++)
    {
        Progression p = progression->at(i);
        fout.write((char*)&p.trainError, sizeof(Float));
        fout.write((char*)&p.testError, sizeof(Float));
        fout.write((char*)&p.alpha, sizeof(Float));
        fout.write((char*)&p.trainCount, sizeof(int));
        fout.write((char*)&p.testCount, sizeof(int));
    }

    fout.close();
}

void RandShuffle(ImageSet set, vector<int> *separation)
{
    int index = 0;
    int num = 0;
    //while (index < set.count)
    //{
    //    for (int i = index; i < set.count; i++)
    //    {
    //        if (set.images[i].num == num)
    //        {
    //            Image p = set.images[i];
    //            set.images[i] = set.images[index];
    //            set.images[index] = p;
    //            index++;
    //        }
    //    }
    //    num++;
    //    separation->push_back(index-1);
    //}
    //for (int i = index; i < set.count; i++)
    //{
    //    if (set.images[i].num == num)
    //    {
    //        Image p = set.images[i];
    //        int n = rand() % set.count;
    //        set.images[i] = set.images[n];
    //        set.images[n] = p;
    //    }
    //}
    for (int i = 0; i < 10000; i++)
    {
        int n1 = rand() % set.count;
        int n2 = rand() % set.count;

        if (n1 == n2)
        {
            i--;
            continue;
        }

        Image p = set.images[n1];
        set.images[n1] = set.images[n2];
        set.images[n2] = p;
    }
}

void FillImages(ImageSet set, Input *inputs, TargetResult *targets)
{
    for (int i = 0; i < set.count; i++)
    {
        net.Initialize(&inputs[i]);
        net.Initialize(&targets[i]);

        for (int j = 0; j < 28; j++)
            for (int k = 0; k < 28; k++)
            {
                unsigned char c1 = set.images[i].pixels[j * 2][k * 2];
                unsigned char c2 = set.images[i].pixels[j * 2 + 1][k * 2];
                unsigned char c3 = set.images[i].pixels[j * 2][k * 2 + 1];
                unsigned char c4 = set.images[i].pixels[j * 2 + 1][k * 2 + 1];
                inputs[i].m_Input(j * 28 + k, 0) = set.images[i].pixels[j][k] / (255.0);
            }
        for (int j = 0; j < 10; j++)
            if (j == set.images[i].num)
                targets[i].m_Target(j) = 1;
            else
                targets[i].m_Target(j) = 0;
    }
}

int TestCount(Input *inputs, TargetResult *targets, int size)
{
    ForwardResult result;

    net.Initialize(&result);
    int correct = 0;
    for (int i = 0; i < size; i++)
    {
        net.Forward(inputs[i], &result);
        Float max = -_FLOAT_MAX;
        int index = -1;
        for (int j = 0; j < 10; j++)
            if (result.m_LayerAs[2].mf[j] > max)
            {
                max = result.m_LayerAs[2].mf[j];
                index = j;
            }

        if (targets[i].m_Target.mf[index] >= 1)
            correct++;
    }
    net.Delete(&result);

    return correct;
}

void Performance(int count, Input *inputs, TargetResult *targets, ForwardResult temp, Float *outError, int *correctCount)
{
    int nCorrect = 0;
    Float error = 0;
    for (int i = 0; i < count; i++)
    {
        net.Forward(inputs[i], &temp);
        error += net.Cost(temp, targets[i]/*, regularization*/);
        {
            Float max = -_FLOAT_MAX;
            int index = -1;
            for (int j = 0; j < 10; j++)
                if (max < temp.m_LayerAs[2].mf[j])
                {
                    max = temp.m_LayerAs[2].mf[j];
                    index = j;
                }
            if (targets[i].m_Target.mf[index] >= 1)
            {
                nCorrect++;
            }
        }
    }

    *correctCount = nCorrect;
    *outError = error / count;
}

void Neural_Network()
{
    ImageSet trainSet, testSet;
    trainSet = ReadImageset("train");
    testSet = ReadImageset("test");

    //return set;

    //NeuralNetwork net;
    int layers[3] = { 28 * 28, 15, 10 };
    //net.CreateFromFile("Weights\\weights.dat");

    //srand(6136);
    //srand(14161);
    srand(981753);
    net.Create(layers, 3);


    srand(519875);

    Input *trainInputs = new Input[trainSet.count];
    TargetResult *trainTargets = new TargetResult[trainSet.count];

    Input *testInputs = new Input[testSet.count];
    TargetResult *testTargets = new TargetResult[testSet.count];

    FillImages(trainSet, trainInputs, trainTargets);
    FillImages(testSet, testInputs, testTargets);

    //int correct = TestCount(inputs, targets, set.count);

    vector<int> separation;
    srand(5172);
    std::vector<Progression> progress;
    //int previous = separation[4];


    TrainingParam param;
    param.m_fAlpha = 3;
    param.m_fLamda = 0;
    param.m_nSampleNum = 1;

    TrainTemp trainTemp;
    net.Initialize(&trainTemp);

    int batchSize = 10;

    Progression __p;
    __p.alpha = 0;
    __p.batchsize = 0;
    __p.start = 0;
    __p.testError = 0;
    __p.trainError = 0;

    ForwardResult forwardResult;
    net.Initialize(&forwardResult);
    Progression p;

    Performance(trainSet.count, trainInputs, trainTargets, forwardResult, &__p.trainError, &__p.trainCount);
    Performance(testSet.count, testInputs, testTargets, forwardResult, &__p.testError, &__p.testCount);

    progress.push_back(__p);

    int progressCount = 0;
    do
    {
        for (int start = 0; start + batchSize <= trainSet.count; start += batchSize)
        {
            RandShuffle(trainSet, &separation);
            net.Train(trainInputs + start, trainTargets + start, batchSize, param, trainTemp);

            p.start = start;
            p.batchsize = batchSize;
            p.alpha = param.m_fAlpha;
            p.testError = __p.testError;
            p.testCount = __p.testCount;

            Performance(batchSize, trainInputs + start, trainTargets + start, forwardResult, &p.trainError, &p.trainCount);

            progress.push_back(p);

            if (progress.size() >= 10001)
            {
                p = progress[progress.size() - 1];
                progress.erase(progress.begin() + 10000);

                SaveProgress("Weights\\progression" + to_string(progressCount) + ".dat", &progress);
                progressCount++;
                progress.clear();
                progress.push_back(p);
            }
        }
        Performance(testSet.count, testInputs, testTargets, forwardResult, &__p.testError, &__p.testCount);

    } while (p.trainError > 0.01);
    //net.SaveToFile("Weights\\weights.dat");
    net.SaveToFile("Weights\\weights.dat");
    SaveProgress("Weights\\progression.dat", &progress);


    for (int i = 0; i < trainSet.count; i++)
    {
        net.Delete(&trainInputs[i]);
        net.Delete(&trainTargets[i]);
    }
    delete[] trainInputs;
    delete[] trainTargets;
    for (int i = 0; i < testSet.count; i++)
    {
        net.Delete(&testInputs[i]);
        net.Delete(&testTargets[i]);
    }
    delete[] testInputs;
    delete[] testTargets;
    net.Delete();
}

int ReadFile(unsigned char *data, int index)
{
    string fileName = "TestData\\train_image.dat";

    ifstream fin(fileName, ios::in | ios::binary | ios::ate);
    ifstream flabel("TestData\\train_label.dat", ios::in | ios::binary | ios::ate);
    fin.seekg(0, ios::beg);
    flabel.seekg(0, ios::beg);

    if (!fin.is_open())
        return 0;
    int magicNumber = 0;
    unsigned char c[4];
    fin.read((char*)c, 4);
    flabel.read((char*)c, 4);
    magicNumber = ((c[0] * 256 + c[1]) * 256 + c[2]) * 256 + c[3];

    int count = 0;
    flabel.read((char*)c, 4);
    fin.read((char*)c, 4);
    count = ((c[0] * 256 + c[1]) * 256 + c[2]) * 256 + c[3];

    int rowNum, columnNum;
    fin.read((char*)c, 4);
    rowNum = ((c[0] * 256 + c[1]) * 256 + c[2]) * 256 + c[3];
    fin.read((char*)c, 4);
    columnNum = ((c[0] * 256 + c[1]) * 256 + c[2]) * 256 + c[3];

    fin.seekg(rowNum * columnNum * index, std::ios::cur);
    fin.read((char*)data, rowNum * columnNum);

    flabel.seekg(index, std::ios::cur);

    flabel.read((char*)(&c), 1);

    return c[0];
}

//void ReadToNet()
//{
//    ifstream fin("weights.dat");
//
//    for (int i = 0; i < 2; i++)
//    {
//        for (int j = 0; j < net.m_LayerCount(i + 1); j++)
//        {
//            for (int k = 0; k < net.m_LayerCount(i); k++)
//            {
//                double D = 0;
//                fin >> D;
//                net.m_pWeights[i](j, k) = D;
//            }
//        }
//    }
//    for (int i = 0; i < 2; i++)
//    {
//        for (int j = 0; j < net.m_LayerCount(i + 1); j++)
//        {
//            double D = 0;
//            fin >> D;
//            net.m_pBiases[i](j) = D;
//        }
//    }
//
//    fin.close();
//}

int wWinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR    lpCmdLine,
    _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);


    g_Strokes = new vector<vector<D2D1_POINT_2F>>();
    //Network::Main();

    //ImageSet set = ReadImageset();
    //NeuralNetwork weights;


    net.CreateFromFile("weights.dat");
    //Neural_Network();

    //if (!net.CreateFromFile("weights.dat"))
    //{
    //    MessageBox(NULL, "Can\'t find file \"weights.dat\"", "Error", MB_ICONERROR);
    //    return 0;
    //}



    //g_bHasAnswer = true;

    if (FAILED(InitD2D()))
    {
        MessageBox(NULL, Text("Failed creating D2D devices!"), Text("ERROR"), 0);
        return FALSE;
    }

    // Initialize global strings
    szTitle = Text("Title");
    szWindowClass = Text("Animation2D");
    MyRegisterClass(hInstance);

    // Perform application initialization:
    if (!InitInstance(hInstance, nCmdShow))
    {
        MessageBox(NULL, Text("Failed creating window!"), Text("ERROR"), 0);
        return FALSE;
    }


    MSG msg;
    ZeroMemory(&msg, sizeof(MSG));

    clock_t t = clock();
    int index = 0;
    // Main message loop:
    while (msg.message != WM_QUIT)
    {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        clock_t newT = clock();

        //if (newT - t > CLOCKS_PER_SEC)
        //{
        //    t = newT;
        //    g_nAnswer = ReadFile(image, index++);
        //    Refresh(g_hWnd);
        //}
    }

    CleanUpD2D();

    //delete[] set.images;
    net.Delete();
    delete g_Strokes;
    _CrtDumpMemoryLeaks();

    return (int)msg.wParam;
}