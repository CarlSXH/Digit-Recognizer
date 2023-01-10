
#include "NeuralNetwork.h"


inline Float RandRange(Float fMin, Float fMax)
{
    return rand() * 3.0518509475997192297128208258309e-5f * (fMax - fMin) + fMin;
}

inline Float SingleDir(Float a, Float b)
{
    return SigmoidDirOutput(b) * a;
}
inline Float Power2(Float a)
{
    return a * a;
}


NeuralNetwork::NeuralNetwork() :
    m_pWeights(0), m_LayerCount()
#ifdef _CPU_RUNTIME_DEBUG
    , m_bDebug(false)
#endif
{
}

void NeuralNetwork::Create(int *rpfLayerCount, int rnCount)
{
    MatInit(&m_LayerCount, rnCount, 1);
    for (int i = 0; i < rnCount; i++)
        m_LayerCount(i) = rpfLayerCount[i];

    m_pWeights = new Matrix[rnCount - 1];
    m_pBiases = new Matrix[rnCount - 1];
    const Float epsilon = 0.0001;
    for (int i = 0; i < rnCount - 1; i++)
    {
        MatInit(&m_pWeights[i], m_LayerCount(i + 1), m_LayerCount(i));
        MatInit(&m_pBiases[i], m_LayerCount(i + 1), 1);
        MatRnd(&m_pWeights[i], -epsilon, epsilon);
        MatRnd(&m_pBiases[i], -epsilon, epsilon);
    }
}

// 4   bits, 32 bit int: magic number (0x0098f8ab) (completely random XD).
// 4   bits, 32 bit int: number of layers (n).
// 4*n bits, 32 bit int: the number of units in layers.
// 1   bit,  8 bit char: 1 (true) or 0 (false), whether it's using float (1) or double (0)
// ... bits, 32 or 64 bit floating number: weights.
// ... bits, 32 or 64 bit floating number: bias.
bool NeuralNetwork::CreateFromFile(string fileName)
{
    ifstream fin(fileName, std::ios::binary);

    if (!fin.is_open())
        return false;

    int magicNum = 0x0098f8ab;
    fin.read((char*)(&magicNum), 4);

    int layerNum;
    fin.read((char*)(&layerNum), 4);

    MatInit(&m_LayerCount, layerNum, 1);
    m_pWeights = new Matrix[layerNum - 1];
    m_pBiases = new Matrix[layerNum - 1];

    for (int i = 0; i < layerNum; i++)
    {
        int unitNum = m_LayerCount(i);
        fin.read((char*)(&unitNum), 4);
        m_LayerCount(i) = unitNum;
    }
    for (int i = 0; i < layerNum - 1; i++)
    {
        MatInit(&m_pWeights[i], m_LayerCount(i + 1), m_LayerCount(i));
        MatInit(&m_pBiases[i], m_LayerCount(i + 1), 1);
    }

    char typeFlag;

    fin.read(&typeFlag, 1);

    int size = sizeof(Float);
    //if (typeFlag == 0)
    //    size = sizeof(double);
    //else if (typeFlag == 1)
    //    size = sizeof(float);

    // temporary variable
    Float f;


    for (int i = 0; i < layerNum - 1; i++)
    {
        for (int j = 0; j < m_pWeights[i].Row; j++)
        {
            for (int k = 0; k < m_pWeights[i].Col; k++)
            {
                fin.read((char*)(&f), size);
                m_pWeights[i](j, k) = f;
            }
        }
    }

    for (int i = 0; i < layerNum - 1; i++)
    {
        for (int j = 0; j < m_pBiases[i].Row; j++)
        {
            fin.read((char*)(&f), size);
            m_pBiases[i](j) = f;
        }
    }

    return true;
}
void NeuralNetwork::SaveToFile(string fileName)
{
    ofstream fout(fileName, std::ios::out | std::ios::binary | std::ios::ate);

    int magicNum = 0x0098f8ab;
    fout.write((char*)(&magicNum), 4);

    int layerNum = m_LayerCount.Row;
    fout.write((char*)(&layerNum), 4);

    for (int i = 0; i < layerNum; i++)
    {
        int unitNum = m_LayerCount(i);
        fout.write((char*)(&unitNum), 4);
    }


    char typeFlag = 0;

#ifdef _USE_FLOAT
    typeFlag = 1;
#endif
#ifdef _USE_DOUBLE
    typeFlag = 0;
#endif

    fout.write(&typeFlag, 1);

    // temporary variable
    Float f;

    for (int i = 0; i < layerNum - 1; i++)
    {
        for (int j = 0; j < m_pWeights[i].Row; j++)
        {
            for (int k = 0; k < m_pWeights[i].Col; k++)
            {
                f = m_pWeights[i](j, k);
                fout.write((char*)(&f), sizeof(Float));
            }
        }
    }

    for (int i = 0; i < layerNum - 1; i++)
    {
        for (int j = 0; j < m_pBiases[i].Row; j++)
        {
            f = m_pBiases[i](j);
            fout.write((char*)(&f), sizeof(Float));
        }
    }
}
void NeuralNetwork::Delete()
{
    if (m_pWeights != NULL)
    {
        for (int i = 0; i < m_LayerCount.Row - 1; i++)
            MatDelete(&m_pWeights[i]);
        delete[] m_pWeights;
        m_pWeights = NULL;
    }
    if (m_pBiases != NULL)
    {
        for (int i = 0; i < m_LayerCount.Row - 1; i++)
            MatDelete(&m_pBiases[i]);
        delete[] m_pBiases;
        m_pBiases = NULL;
    }
    MatDelete(&m_LayerCount);
}

void NeuralNetwork::Initialize(ForwardResult *result)
{
    result->m_nCount = m_LayerCount.Row;
    result->m_LayerAs = new Matrix[result->m_nCount];
    result->m_LayerZs = new Matrix[result->m_nCount - 1];

    for (int i = 0; i < result->m_nCount; i++)
        MatInit(&result->m_LayerAs[i], m_LayerCount(i), 1);
    for (int i = 0; i < result->m_nCount - 1; i++)
        MatInit(&result->m_LayerZs[i], m_LayerCount(i + 1), 1);
}
void NeuralNetwork::Initialize(Input *input)
{
    MatInit(&input->m_Input, m_LayerCount(0), 1);
}
void NeuralNetwork::Initialize(TargetResult *target)
{
    MatInit(&target->m_Target, m_LayerCount(m_LayerCount.Row - 1), 1);
}
void NeuralNetwork::Initialize(ResultDerivative *derivative)
{
    derivative->m_pDWeights = new Matrix[m_LayerCount.Row - 1];
    derivative->m_pDBiases = new Matrix[m_LayerCount.Row - 1];

    for (int i = 0; i < m_LayerCount.Row - 1; i++)
    {
        MatInit(&derivative->m_pDWeights[i], m_pWeights[i].Row, m_pWeights[i].Col);
        MatInit(&derivative->m_pDBiases[i], m_pBiases[i].Row, 1);
    }
}
void NeuralNetwork::Initialize(DerivativeTemp *temp)
{
    temp->m_Errors = new Matrix[m_LayerCount.Row - 1];
    temp->m_ErrorsT = new Matrix[m_LayerCount.Row - 2];

    for (int i = 1; i < m_LayerCount.Row; i++)
        MatInit(&temp->m_Errors[i - 1], m_LayerCount(i), 1);
    for (int i = 1; i < m_LayerCount.Row - 1; i++)
        MatInit(&temp->m_ErrorsT[i - 1], m_LayerCount(i), m_LayerCount(i + 1));

}
void NeuralNetwork::Initialize(GradientCheckResult *numerical)
{
}
void NeuralNetwork::Initialize(TrainTemp *temp)
{
    Initialize(&temp->m_AccuDerivative);

    Initialize(&temp->m_ForwardResult);
    Initialize(&temp->m_Derivative);
    Initialize(&temp->m_Temp);

}
void NeuralNetwork::Delete(ForwardResult *result)
{
    if (result->m_LayerAs != NULL)
    {
        for (int i = 0; i < m_LayerCount.Row; i++)
            MatDelete(&result->m_LayerAs[i]);
        delete[] result->m_LayerAs;
        result->m_LayerAs = NULL;
    }
    if (result->m_LayerZs != NULL)
    {
        for (int i = 0; i < m_LayerCount.Row - 1; i++)
            MatDelete(&result->m_LayerZs[i]);
        delete[] result->m_LayerZs;
        result->m_LayerZs = NULL;
    }
    result->m_nCount = 0;
}
void NeuralNetwork::Delete(Input *input)
{
    MatDelete(&input->m_Input);
}
void NeuralNetwork::Delete(TargetResult *target)
{
    MatDelete(&target->m_Target);
}
void NeuralNetwork::Delete(ResultDerivative *derivative)
{
    if (derivative->m_pDWeights != NULL)
    {
        for (int i = 0; i < m_LayerCount.Row - 1; i++)
            MatDelete(&derivative->m_pDWeights[i]);
        delete[] derivative->m_pDWeights;
        derivative->m_pDWeights = NULL;
    }
    if (derivative->m_pDBiases != NULL)
    {
        for (int i = 0; i < m_LayerCount.Row - 1; i++)
            MatDelete(&derivative->m_pDBiases[i]);
        delete[] derivative->m_pDBiases;
        derivative->m_pDBiases = NULL;
    }
}
void NeuralNetwork::Delete(DerivativeTemp *temp)
{
    if (temp->m_Errors != NULL)
    {
        for (int i = 0; i < m_LayerCount.Row - 1; i++)
            MatDelete(&temp->m_Errors[i]);
        delete[] temp->m_Errors;
        temp->m_Errors = NULL;
    }
    if (temp->m_ErrorsT != NULL)
    {
        for (int i = 0; i < m_LayerCount.Row - 2; i++)
            MatDelete(&temp->m_ErrorsT[i]);
        delete[] temp->m_ErrorsT;
        temp->m_ErrorsT = NULL;
    }
}
void NeuralNetwork::Delete(GradientCheckResult *numerical)
{
}
void NeuralNetwork::Delete(TrainTemp *temp)
{
    Delete(&temp->m_AccuDerivative);
    Delete(&temp->m_ForwardResult);
    Delete(&temp->m_Derivative);
    Delete(&temp->m_Temp);
}

void NeuralNetwork::Forward(Input input, ForwardResult *result, int n)
{
    result->m_nCount = m_LayerCount.Row;
    MatCpy(&result->m_LayerAs[n], &input.m_Input);

    for (int i = n + 1; i < m_LayerCount.Row; i++)
    {
#ifdef _CPU_DEBUG
#ifdef _CPU_RUNTIME_DEBUG
        if (m_bDebug)
#endif
        {
            OutputDebugString("The weights of this layer:\n");
            MatDebug(&m_pWeights[i - 1]);
            OutputDebugString("\n");

            OutputDebugString("The output of the last layer:\n");
            MatDebug(&result->m_LayerAs[i - 1]);
            OutputDebugString("\n");

            OutputDebugString("Calculating the z values...\n");
        }
#endif
        MatDot(&result->m_LayerZs[i - 1], &m_pWeights[i - 1], &result->m_LayerAs[i - 1]);

#ifdef _CPU_DEBUG
#ifdef _CPU_RUNTIME_DEBUG
        if (m_bDebug)
#endif
        {
            OutputDebugString("\nThe z values:\n");
            MatDebug(&result->m_LayerZs[i - 1]);
            OutputDebugString("\n");

            OutputDebugString("The biases of this layer:\n");
            MatDebug(&m_pBiases[i - 1]);
            OutputDebugString("\n");

            OutputDebugString("Calculating the a values...\n");
        }
#endif
        MatAdd(&result->m_LayerZs[i - 1], &result->m_LayerZs[i - 1], &m_pBiases[i - 1]);
        MatApply(&result->m_LayerAs[i], &result->m_LayerZs[i - 1], Sigmoid);


#ifdef _CPU_DEBUG
#ifdef _CPU_RUNTIME_DEBUG
        if (m_bDebug)
#endif
        {
            OutputDebugString("\nThe a values:\n");
            MatDebug(&result->m_LayerAs[i]);
            OutputDebugString("\n");
        }
#endif

    }
}
Float NeuralNetwork::Cost(ForwardResult result, TargetResult target/*, Float sumSqr*/)
{
    Matrix x = result.m_LayerAs[result.m_nCount - 1];
    Matrix y = target.m_Target;
    Float sum = 0;

    for (int i = 0; i < x.Row; i++)
    {
        //sum += Power2(x(i) - y(i));
        if (y(i) == 0)
            sum += log(1 - x(i));
        else
            sum += log(x(i));
    }

    sum = -sum;

    return sum/* + sumSqr*/;
}
Float NeuralNetwork::SumSqr()
{
    Float sum = 0;
    for (int i = 0; i < m_LayerCount.Row - 1; i++)
    {
        Float sumSqr;
        MatApply(&sumSqr, &m_pWeights[i], Power2);
        sum += sumSqr;
    }
    return sum;
}

void NeuralNetwork::Derivative(ForwardResult result, TargetResult target, DerivativeTemp temp, ResultDerivative *derivative)
{
    int count = m_LayerCount.Row;
    Matrix *errors = temp.m_Errors;


#ifdef _CPU_DEBUG
#ifdef _CPU_RUNTIME_DEBUG
    if (m_bDebug)
#endif
    {
        OutputDebugString("The forward result:\n");
        MatDebug(&result.m_LayerAs[count - 1]);
        OutputDebugString("\n");

        OutputDebugString("The expected result:\n");
        MatDebug(&target.m_Target);
        OutputDebugString("\n");

        OutputDebugString("Calculating output error...\n");
    }
#endif


    MatSub(&errors[count - 2], &result.m_LayerAs[result.m_nCount - 1], &target.m_Target);

#ifdef _CPU_DEBUG
#ifdef _CPU_RUNTIME_DEBUG
    if (m_bDebug)
#endif
    {
        OutputDebugString("\nThe error of the output layer:\n");
        MatDebug(&errors[count - 2]);
        OutputDebugString("\n");
    }
#endif
    for (int i = count - 3; i >= 0; i--)
    {


#ifdef _CPU_DEBUG
#ifdef _CPU_RUNTIME_DEBUG
        if (m_bDebug)
#endif
        {
            OutputDebugString("The error of the previous layer:\n");
            MatDebug(&errors[i + 1]);
            OutputDebugString("\n");

            OutputDebugString("The weights of the layers:\n");
            MatDebug(&m_pWeights[i + 1]);
            OutputDebugString("\n");

            OutputDebugString("The the z value of the layer:\n");
            MatDebug(&result.m_LayerZs[i]);
            OutputDebugString("\n");

            OutputDebugString("Calculating error of this layer..\n");
        }
#endif
        MatT(&temp.m_ErrorsT[i], &m_pWeights[i + 1]);
        MatDot(&errors[i], &temp.m_ErrorsT[i], &errors[i + 1]);

#ifdef _CPU_DEBUG
#ifdef _CPU_RUNTIME_DEBUG
        if (m_bDebug)
#endif
        {
            OutputDebugString("/nIntermediate value:\n");
            MatDebug(&errors[i]);
            OutputDebugString("\n");

        }
#endif

        MatApply(&errors[i], &errors[i], &result.m_LayerAs[i + 1], SingleDir);

#ifdef _CPU_DEBUG
#ifdef _CPU_RUNTIME_DEBUG
        if (m_bDebug)
#endif
        {
            OutputDebugString("The the error of this layer:\n");
            MatDebug(&errors[i]);
            OutputDebugString("\n");
        }
#endif
    }

    for (int i = 0; i < count - 1; i++)
    {

#ifdef _CPU_DEBUG
#ifdef _CPU_RUNTIME_DEBUG
        if (m_bDebug)
#endif
        {
            OutputDebugString("The the error of this layer:\n");
            MatDebug(&errors[i]);
            OutputDebugString("\n");

            OutputDebugString("The the a value of this layer:\n");
            MatDebug(&result.m_LayerAs[i]);
            OutputDebugString("\n");

            OutputDebugString("Calculating the derivative for this layer:\n");
        }
#endif

        for (int j = 0; j < errors[i].Row; j++)
            for (int k = 0; k < result.m_LayerAs[i].Row; k++)
                derivative->m_pDWeights[i](j, k) = errors[i](j, 0) * result.m_LayerAs[i](k, 0);
        MatCpy(&derivative->m_pDBiases[i], &errors[i]);

#ifdef _CPU_DEBUG
#ifdef _CPU_RUNTIME_DEBUG
        if (m_bDebug)
#endif
        {
            OutputDebugString("\nThe the derivative of this layer:\n");
            MatDebug(&derivative->m_pDWeights[i]);
            OutputDebugString("\n");
        }
#endif
    }
}


void NeuralNetwork::NumericalGradientCheck(Input input, TargetResult target, ResultDerivative derivative, GradientCheckResult *numerical, TrainingParam param)
{
    Float epsilon = 0.0001;
    Float sum = 0.0;
    Float max = -_FLOAT_MAX, min = _FLOAT_MAX;

    ForwardResult result;

    Initialize(&result);
    Float regularization = (param.m_fLamda * SumSqr()) / (2 * param.m_nSampleNum);

    for (int i = 0; i < m_LayerCount.Row - 1; i++)
    {
        for (int j = 0; j < m_pWeights[i].Row; j++)
        {
            for (int k = 0; k < m_pWeights[i].Col; k++)
            {
                Float dir = 0.0;

                m_pWeights[i](j, k) += epsilon;
                Forward(input, &result);
                dir = Cost(result, target/*, regularization*/);

                m_pWeights[i](j, k) -= 2 * epsilon;
                Forward(input, &result);
                dir = dir - Cost(result, target/*, regularization*/);

                m_pWeights[i](j, k) += epsilon;

                dir = dir / (epsilon * 2);

                Float dif = dir - derivative.m_pDWeights[i](j, k);
                if (dif > max)
                    max = dif;
                if (dif < min)
                    min = dif;
                sum += dif;
            }
        }
    }

    for (int i = 0; i < m_LayerCount.Row - 1; i++)
    {
        for (int j = 0; j < m_pBiases[i].Row; j++)
        {
            Float dir = 0.0;

            m_pBiases[i](j) += epsilon;
            Forward(input, &result);
            dir = Cost(result, target/*, regularization*/);

            m_pBiases[i](j) -= 2 * epsilon;
            Forward(input, &result);
            dir = dir - Cost(result, target/*, regularization*/);

            m_pBiases[i](j) += epsilon;

            dir = dir / (epsilon * 2);

            Float dif = dir - derivative.m_pDBiases[i](j);
            if (dif > max)
                max = dif;
            if (dif < min)
                min = dif;
            sum += dif;
        }
    }

    Delete(&result);

    int count = 0;

    for (int i = 0; i < m_LayerCount.Row - 1; i++)
    {
        count += m_pWeights[i].Row * m_pWeights[i].Col;
        count += m_pBiases[i].Row;
    }

    numerical->m_dMax = max;
    numerical->m_dMin = min;
    numerical->m_dSum = sum;
    numerical->m_dAverage = sum / count;
}

void NeuralNetwork::Train(Input *inputs, TargetResult *targets, int count, TrainingParam param, TrainTemp trainTemp)
{
    ResultDerivative accuDerivative = trainTemp.m_AccuDerivative;

    ForwardResult forwardResult = trainTemp.m_ForwardResult;
    ResultDerivative derivative = trainTemp.m_Derivative;
    DerivativeTemp temp = trainTemp.m_Temp;

    //Float cost = 0;
    int iteration = 0;

    Float alphaTemp = 1 / (Float)count * param.m_fAlpha;
    //Float lamdaTemp = 1 - param.m_fLamda;

    for (int i = 0; i < m_LayerCount.Row - 1; i++)
    {
        MatClear(&accuDerivative.m_pDWeights[i], (Float)0);
        MatClear(&accuDerivative.m_pDBiases[i], (Float)0);
    }

    //do
    //{
    //Float regularization = 0;
    //if (param.m_fLamda != 0)
    //    regularization = (param.m_fLamda * SumSqr()) / (2 * param.m_nSampleNum);

    for (int i = 0; i < m_LayerCount.Row - 1; i++)
    {
        MatClear(&accuDerivative.m_pDWeights[i], (Float)0);
        MatClear(&accuDerivative.m_pDBiases[i], (Float)0);
    }

    for (int i = 0; i < count; i++)
    {
        Forward(inputs[i], &forwardResult);
        Derivative(forwardResult, targets[i], temp, &derivative);
        //cost += Cost(forwardResult, targets[i]/*, regularization*/);

        for (int j = 0; j < m_LayerCount.Row - 1; j++)
        {
            MatAdd(&accuDerivative.m_pDWeights[j], &accuDerivative.m_pDWeights[j], &derivative.m_pDWeights[j]);
            MatAdd(&accuDerivative.m_pDBiases[j], &accuDerivative.m_pDBiases[j], &derivative.m_pDBiases[j]);
        }
    }


    for (int i = 0; i < m_LayerCount.Row - 1; i++)
    {
        MatMul(&accuDerivative.m_pDWeights[i], &accuDerivative.m_pDWeights[i], alphaTemp);
        MatMul(&accuDerivative.m_pDBiases[i], &accuDerivative.m_pDBiases[i], alphaTemp);

        //MatMul(&m_pWeights[i], &m_pWeights[i], lamdaTemp);

        MatSub(&m_pWeights[i], &m_pWeights[i], &accuDerivative.m_pDWeights[i]);
        MatSub(&m_pBiases[i], &m_pBiases[i], &accuDerivative.m_pDBiases[i]);
    }

    //cost /= count;

    //Progression p;
    //p.trainError = cost;
    //p.start = start;
    //p.batchsize = count;
    //p.alpha = param.m_fAlpha;
    //iteration++;
    //} while (cost >= 0.01 && iteration < maxIteration);

    //return p;
}


void __Main()
{
    NeuralNetwork net;
    int layers[5] = { 28 * 28, 15, 10 };
    net.Create(layers, 3);

    Input v;

    net.Initialize(&v);

    MatRnd(&v.m_Input, (Float)0.0, (Float)1.0);

    ForwardResult result;
    net.Initialize(&result);

    net.Forward(v, &result);

    TargetResult target;
    net.Initialize(&target);

    MatRnd(&target.m_Target, (Float)0.0, (Float)1.0);

    for (int i = 0; i < 10; i++)
    {
        target.m_Target(i) = RandRange(0, 2) > 1.0 ? 1 : 0;
    }

    ResultDerivative derivative;
    net.Initialize(&derivative);

    DerivativeTemp temp;
    net.Initialize(&temp);

    for (int i = 0; i < 1000; i++)
        net.Forward(v, &result);

    GradientCheckResult numerical;
    TrainingParam param;
    param.m_fAlpha = 0.05;
    param.m_fLamda = 0.1;
    param.m_nSampleNum = 1;

    net.NumericalGradientCheck(v, target, derivative, &numerical, param);

    net.Delete(&result);
    net.Delete(&target);
    net.Delete(&derivative);
    net.Delete(&temp);

    net.Delete();

    return;
}