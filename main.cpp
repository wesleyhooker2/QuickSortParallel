#include <cstdlib>
#include "immintrin.h"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <algorithm> // std::sort
#include <vector>    // std::vector
#include <math.h>
#include <iterator>
#include <unistd.h>
#include <omp.h>
#include <random>
#include <climits>
#define N 33'554'432 // 2^25
#define NUM_PER_THREAD 65536
#define NUM_PER_SORTED 16384
using namespace std;

//Globals
const int ARRAY_LENGTH = N;
//Globals

void printVectorInt(__m512i v, string name)
{
#if defined(__GNUC__)
    int *temp = (int *)aligned_alloc(64, sizeof(int) * 16);
#elif defined(_MSC_VER)
    int *temp = (int *)_aligned_malloc(sizeof(int) * 16, 64);
#endif
    _mm512_store_si512(temp, v);
    printf("The vector called %s contains: ", name.c_str());
    for (int i = 0; i < 16; i++)
    {
        printf("%02d ", temp[i]);
    }
    printf("\n");
#if defined(__GNUC__)
    free(temp);
#elif defined(_MSC_VER)
    _aligned_free(temp);
#endif
}

void printArray(int v[], int numElements, string name)
{
    for (int i = 0; i < numElements; i++)
    {
        if (i % 16 == 0)
        {
            cout << endl;
        }
        cout << " " << v[i];
    }
    cout << endl;
}

void bitonic_sort(__m512i &A1in, __m512i &A2in, __m512i &B1in, __m512i &B2in, __m512i &C1in, __m512i &C2in, __m512i &D1in, __m512i &D2in,
                  __m512i &A1out, __m512i &A2out, __m512i &B1out, __m512i &B2out, __m512i &C1out, __m512i &C2out, __m512i &D1out, __m512i &D2out)
{
    //reverse A2in
    A2in = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), A2in);
    B2in = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), B2in);
    C2in = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), C2in);
    D2in = _mm512_permutexvar_epi32(_mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), D2in);

    //L1 Start
    __m512i MinA = _mm512_min_epi32(A1in, A2in);
    __m512i MaxA = _mm512_max_epi32(A1in, A2in);
    __m512i MinB = _mm512_min_epi32(B1in, B2in);
    __m512i MaxB = _mm512_max_epi32(B1in, B2in);
    __m512i MinC = _mm512_min_epi32(C1in, C2in);
    __m512i MaxC = _mm512_max_epi32(C1in, C2in);
    __m512i MinD = _mm512_min_epi32(D1in, D2in);
    __m512i MaxD = _mm512_max_epi32(D1in, D2in);

    A1in = _mm512_permutex2var_epi32(MinA, _mm512_set_epi32(23, 22, 21, 20, 19, 18, 17, 16, 7, 6, 5, 4, 3, 2, 1, 0), MaxA);
    A2in = _mm512_permutex2var_epi32(MinA, _mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 15, 14, 13, 12, 11, 10, 9, 8), MaxA);
    B1in = _mm512_permutex2var_epi32(MinB, _mm512_set_epi32(23, 22, 21, 20, 19, 18, 17, 16, 7, 6, 5, 4, 3, 2, 1, 0), MaxB);
    B2in = _mm512_permutex2var_epi32(MinB, _mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 15, 14, 13, 12, 11, 10, 9, 8), MaxB);
    C1in = _mm512_permutex2var_epi32(MinC, _mm512_set_epi32(23, 22, 21, 20, 19, 18, 17, 16, 7, 6, 5, 4, 3, 2, 1, 0), MaxC);
    C2in = _mm512_permutex2var_epi32(MinC, _mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 15, 14, 13, 12, 11, 10, 9, 8), MaxC);
    D1in = _mm512_permutex2var_epi32(MinD, _mm512_set_epi32(23, 22, 21, 20, 19, 18, 17, 16, 7, 6, 5, 4, 3, 2, 1, 0), MaxD);
    D2in = _mm512_permutex2var_epi32(MinD, _mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 15, 14, 13, 12, 11, 10, 9, 8), MaxD);

    //L2 Start
    MinA = _mm512_min_epi32(A1in, A2in);
    MaxA = _mm512_max_epi32(A1in, A2in);
    MinB = _mm512_min_epi32(B1in, B2in);
    MaxB = _mm512_max_epi32(B1in, B2in);
    MinC = _mm512_min_epi32(C1in, C2in);
    MaxC = _mm512_max_epi32(C1in, C2in);
    MinD = _mm512_min_epi32(D1in, D2in);
    MaxD = _mm512_max_epi32(D1in, D2in);

    A1in = _mm512_permutex2var_epi32(MinA, _mm512_set_epi32(27, 26, 25, 24, 11, 10, 9, 8, 19, 18, 17, 16, 3, 2, 1, 0), MaxA);
    A2in = _mm512_permutex2var_epi32(MinA, _mm512_set_epi32(31, 30, 29, 28, 15, 14, 13, 12, 23, 22, 21, 20, 7, 6, 5, 4), MaxA);
    B1in = _mm512_permutex2var_epi32(MinB, _mm512_set_epi32(27, 26, 25, 24, 11, 10, 9, 8, 19, 18, 17, 16, 3, 2, 1, 0), MaxB);
    B2in = _mm512_permutex2var_epi32(MinB, _mm512_set_epi32(31, 30, 29, 28, 15, 14, 13, 12, 23, 22, 21, 20, 7, 6, 5, 4), MaxB);
    C1in = _mm512_permutex2var_epi32(MinC, _mm512_set_epi32(27, 26, 25, 24, 11, 10, 9, 8, 19, 18, 17, 16, 3, 2, 1, 0), MaxC);
    C2in = _mm512_permutex2var_epi32(MinC, _mm512_set_epi32(31, 30, 29, 28, 15, 14, 13, 12, 23, 22, 21, 20, 7, 6, 5, 4), MaxC);
    D1in = _mm512_permutex2var_epi32(MinD, _mm512_set_epi32(27, 26, 25, 24, 11, 10, 9, 8, 19, 18, 17, 16, 3, 2, 1, 0), MaxD);
    D2in = _mm512_permutex2var_epi32(MinD, _mm512_set_epi32(31, 30, 29, 28, 15, 14, 13, 12, 23, 22, 21, 20, 7, 6, 5, 4), MaxD);

    //L3 Start
    MinA = _mm512_min_epi32(A1in, A2in);
    MaxA = _mm512_max_epi32(A1in, A2in);
    MinB = _mm512_min_epi32(B1in, B2in);
    MaxB = _mm512_max_epi32(B1in, B2in);
    MinC = _mm512_min_epi32(C1in, C2in);
    MaxC = _mm512_max_epi32(C1in, C2in);
    MinD = _mm512_min_epi32(D1in, D2in);
    MaxD = _mm512_max_epi32(D1in, D2in);

    A1in = _mm512_permutex2var_epi32(MinA, _mm512_set_epi32(29, 28, 13, 12, 25, 24, 9, 8, 21, 20, 5, 4, 17, 16, 1, 0), MaxA);
    A2in = _mm512_permutex2var_epi32(MinA, _mm512_set_epi32(31, 30, 15, 14, 27, 26, 11, 10, 23, 22, 7, 6, 19, 18, 3, 2), MaxA);
    B1in = _mm512_permutex2var_epi32(MinB, _mm512_set_epi32(29, 28, 13, 12, 25, 24, 9, 8, 21, 20, 5, 4, 17, 16, 1, 0), MaxB);
    B2in = _mm512_permutex2var_epi32(MinB, _mm512_set_epi32(31, 30, 15, 14, 27, 26, 11, 10, 23, 22, 7, 6, 19, 18, 3, 2), MaxB);
    C1in = _mm512_permutex2var_epi32(MinC, _mm512_set_epi32(29, 28, 13, 12, 25, 24, 9, 8, 21, 20, 5, 4, 17, 16, 1, 0), MaxC);
    C2in = _mm512_permutex2var_epi32(MinC, _mm512_set_epi32(31, 30, 15, 14, 27, 26, 11, 10, 23, 22, 7, 6, 19, 18, 3, 2), MaxC);
    D1in = _mm512_permutex2var_epi32(MinD, _mm512_set_epi32(29, 28, 13, 12, 25, 24, 9, 8, 21, 20, 5, 4, 17, 16, 1, 0), MaxD);
    D2in = _mm512_permutex2var_epi32(MinD, _mm512_set_epi32(31, 30, 15, 14, 27, 26, 11, 10, 23, 22, 7, 6, 19, 18, 3, 2), MaxD);

    //L4 Start
    MinA = _mm512_min_epi32(A1in, A2in);
    MaxA = _mm512_max_epi32(A1in, A2in);
    MinB = _mm512_min_epi32(B1in, B2in);
    MaxB = _mm512_max_epi32(B1in, B2in);
    MinC = _mm512_min_epi32(C1in, C2in);
    MaxC = _mm512_max_epi32(C1in, C2in);
    MinD = _mm512_min_epi32(D1in, D2in);
    MaxD = _mm512_max_epi32(D1in, D2in);

    A1in = _mm512_permutex2var_epi32(MinA, _mm512_set_epi32(30, 14, 28, 12, 26, 10, 24, 8, 22, 6, 20, 4, 18, 2, 16, 0), MaxA);
    A2in = _mm512_permutex2var_epi32(MinA, _mm512_set_epi32(31, 15, 29, 13, 27, 11, 25, 9, 23, 7, 21, 5, 19, 3, 17, 1), MaxA);
    B1in = _mm512_permutex2var_epi32(MinB, _mm512_set_epi32(30, 14, 28, 12, 26, 10, 24, 8, 22, 6, 20, 4, 18, 2, 16, 0), MaxB);
    B2in = _mm512_permutex2var_epi32(MinB, _mm512_set_epi32(31, 15, 29, 13, 27, 11, 25, 9, 23, 7, 21, 5, 19, 3, 17, 1), MaxB);
    C1in = _mm512_permutex2var_epi32(MinC, _mm512_set_epi32(30, 14, 28, 12, 26, 10, 24, 8, 22, 6, 20, 4, 18, 2, 16, 0), MaxC);
    C2in = _mm512_permutex2var_epi32(MinC, _mm512_set_epi32(31, 15, 29, 13, 27, 11, 25, 9, 23, 7, 21, 5, 19, 3, 17, 1), MaxC);
    D1in = _mm512_permutex2var_epi32(MinD, _mm512_set_epi32(30, 14, 28, 12, 26, 10, 24, 8, 22, 6, 20, 4, 18, 2, 16, 0), MaxD);
    D2in = _mm512_permutex2var_epi32(MinD, _mm512_set_epi32(31, 15, 29, 13, 27, 11, 25, 9, 23, 7, 21, 5, 19, 3, 17, 1), MaxD);

    //L5 Start
    MinA = _mm512_min_epi32(A1in, A2in);
    MaxA = _mm512_max_epi32(A1in, A2in);
    MinB = _mm512_min_epi32(B1in, B2in);
    MaxB = _mm512_max_epi32(B1in, B2in);
    MinC = _mm512_min_epi32(C1in, C2in);
    MaxC = _mm512_max_epi32(C1in, C2in);
    MinD = _mm512_min_epi32(D1in, D2in);
    MaxD = _mm512_max_epi32(D1in, D2in);

    A1out = _mm512_permutex2var_epi32(MinA, _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0), MaxA);
    A2out = _mm512_permutex2var_epi32(MinA, _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8), MaxA);
    B1out = _mm512_permutex2var_epi32(MinB, _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0), MaxB);
    B2out = _mm512_permutex2var_epi32(MinB, _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8), MaxB);
    C1out = _mm512_permutex2var_epi32(MinC, _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0), MaxC);
    C2out = _mm512_permutex2var_epi32(MinC, _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8), MaxC);
    D1out = _mm512_permutex2var_epi32(MinD, _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0), MaxD);
    D2out = _mm512_permutex2var_epi32(MinD, _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8), MaxD);
}

void sort_32(int *a, int indexStart)
{
    __m512i A1out, A2out, A1in, A2in, B1in, B1out, B2in, B2out,
        C1in, C1out, C2in, C2out, D1in, D1out, D2in, D2out;

    //Sort first 32
    A1in = _mm512_load_epi32(a + indexStart);
    A2in = _mm512_load_epi32(a + indexStart + 16);
    B1in = _mm512_load_epi32(a + indexStart + 32);
    B2in = _mm512_load_epi32(a + indexStart + 48);
    C1in = _mm512_load_epi32(a + indexStart + 64);
    C2in = _mm512_load_epi32(a + indexStart + 80);
    D1in = _mm512_load_epi32(a + indexStart + 96);
    D2in = _mm512_load_epi32(a + indexStart + 112);
    bitonic_sort(A1in, A2in, B1in, B2in, C1in, C2in, D1in, D2in,
                 A1out, A2out, B1out, B2out, C1out, C2out, D1out, D2out);

    //Write back out to output array
    _mm512_store_si512(a + indexStart, A1out);
    _mm512_store_si512(a + indexStart + 16, A2out);
    _mm512_store_si512(a + indexStart + 32, B1out);
    _mm512_store_si512(a + indexStart + 48, B2out);
    _mm512_store_si512(a + indexStart + 64, C1out);
    _mm512_store_si512(a + indexStart + 80, C2out);
    _mm512_store_si512(a + indexStart + 96, D1out);
    _mm512_store_si512(a + indexStart + 112, D2out);
}

void merge_blocks(int *a, int *output, int mergeBlockSize, int indexStart)
{
    //COPY
    // int *temp_a = (int *)aligned_alloc(64, sizeof(int) * ARRAY_LENGTH);
    // for (int i = 0; i < ARRAY_LENGTH; i++)
    // {
    //     temp_a[i] = a[i];
    // }

    // cout << "\n\nTEMP A";
    // printArray(temp_a, ARRAY_LENGTH, "temp_a");

    __m512i A1out, A2out, A1in, A2in, B1in, B1out, B2in, B2out,
        C1in, C1out, C2in, C2out, D1in, D1out, D2in, D2out;
    int currentBlockSize = mergeBlockSize / 2;
    int numRuns = (mergeBlockSize / 16) - 1 - 2;
    int AIndex1 = (currentBlockSize * 0), AIndex2 = (currentBlockSize * 1);
    int BIndex1 = (currentBlockSize * 2), BIndex2 = (currentBlockSize * 3);
    int CIndex1 = (currentBlockSize * 4), CIndex2 = (currentBlockSize * 5);
    int DIndex1 = (currentBlockSize * 6), DIndex2 = (currentBlockSize * 7);

    //first 16
    A1in = _mm512_load_epi32(a + indexStart + AIndex1);
    A2in = _mm512_load_epi32(a + indexStart + AIndex2);
    B1in = _mm512_load_epi32(a + indexStart + BIndex1);
    B2in = _mm512_load_epi32(a + indexStart + BIndex2);
    C1in = _mm512_load_epi32(a + indexStart + CIndex1);
    C2in = _mm512_load_epi32(a + indexStart + CIndex2);
    D1in = _mm512_load_epi32(a + indexStart + DIndex1);
    D2in = _mm512_load_epi32(a + indexStart + DIndex2);
    AIndex1 += 16;
    AIndex2 += 16;
    BIndex1 += 16;
    BIndex2 += 16;
    CIndex1 += 16;
    CIndex2 += 16;
    DIndex1 += 16;
    DIndex2 += 16;

    bitonic_sort(A1in, A2in, B1in, B2in, C1in, C2in, D1in, D2in,
                 A1out, A2out, B1out, B2out, C1out, C2out, D1out, D2out);
    _mm512_store_si512(output + indexStart + (mergeBlockSize * 0), A1out);
    _mm512_store_si512(output + indexStart + (mergeBlockSize * 1), B1out);
    _mm512_store_si512(output + indexStart + (mergeBlockSize * 2), C1out);
    _mm512_store_si512(output + indexStart + (mergeBlockSize * 3), D1out);

    //Middle 16s
    for (int i = 0; i < numRuns; i++)
    {
        //first input = unused output
        A1in = A2out;
        B1in = B2out;
        C1in = C2out;
        D1in = D2out;

        //second input = unused output
        if (AIndex1 != currentBlockSize && AIndex2 != mergeBlockSize)
        {
            if (a[AIndex1 + indexStart] < a[AIndex2 + indexStart]) //second input = lower value arrays index value
            {
                A2in = _mm512_load_epi32(a + indexStart + AIndex1);
                AIndex1 += 16;
            }
            else
            {
                A2in = _mm512_load_epi32(a + indexStart + AIndex2);
                AIndex2 += 16;
            }
        }
        else
        {
            if (AIndex1 == currentBlockSize)
            {
                A2in = _mm512_load_epi32(a + indexStart + AIndex2);
                AIndex2 += 16;
            }
            else //(blockIndex2 == mergeBlockSize)
            {
                A2in = _mm512_load_epi32(a + indexStart + AIndex1);
                AIndex1 += 16;
            }
        }

        //second input = unused output
        if (BIndex1 != currentBlockSize + (mergeBlockSize * 1) && BIndex2 != mergeBlockSize + (mergeBlockSize * 1))
        {
            if (a[BIndex1 + indexStart] < a[BIndex2 + indexStart]) //second input = lower value arrays index value
            {
                B2in = _mm512_load_epi32(a + BIndex1 + indexStart);
                BIndex1 += 16;
            }
            else
            {
                B2in = _mm512_load_epi32(a + BIndex2 + indexStart);
                BIndex2 += 16;
            }
        }
        else
        {
            if (BIndex1 == currentBlockSize + (mergeBlockSize * 1))
            {
                B2in = _mm512_load_epi32(a + BIndex2 + indexStart);
                BIndex2 += 16;
            }
            else //(blockIndex2 == mergeBlockSize)
            {
                B2in = _mm512_load_epi32(a + BIndex1 + indexStart);
                BIndex1 += 16;
            }
        }

        //second input = unused output
        if (CIndex1 != currentBlockSize + (mergeBlockSize * 2) && CIndex2 != mergeBlockSize + (mergeBlockSize * 2))
        {
            if (a[CIndex1 + indexStart] < a[CIndex2 + indexStart]) //second input = lower value arrays index value
            {
                C2in = _mm512_load_epi32(a + CIndex1 + indexStart);
                CIndex1 += 16;
            }
            else
            {
                C2in = _mm512_load_epi32(a + CIndex2 + indexStart);
                CIndex2 += 16;
            }
        }
        else
        {
            if (CIndex1 == currentBlockSize + (mergeBlockSize * 2))
            {
                C2in = _mm512_load_epi32(a + CIndex2 + indexStart);
                CIndex2 += 16;
            }
            else //(blockIndex2 == mergeBlockSize)
            {
                C2in = _mm512_load_epi32(a + CIndex1 + indexStart);
                CIndex1 += 16;
            }
        }

        //second input = unused output
        if (DIndex1 != currentBlockSize + (mergeBlockSize * 3) && DIndex2 != mergeBlockSize + (mergeBlockSize * 3))
        {
            if (a[DIndex1 + indexStart] < a[DIndex2 + indexStart]) //second input = lower value arrays index value
            {
                D2in = _mm512_load_epi32(a + DIndex1 + indexStart);
                DIndex1 += 16;
            }
            else
            {
                D2in = _mm512_load_epi32(a + DIndex2 + indexStart);
                DIndex2 += 16;
            }
        }
        else
        {
            if (DIndex1 == currentBlockSize + (mergeBlockSize * 3))
            {
                D2in = _mm512_load_epi32(a + DIndex2 + indexStart);
                DIndex2 += 16;
            }
            else //(blockIndex2 == mergeBlockSize)
            {
                D2in = _mm512_load_epi32(a + DIndex1 + indexStart);
                DIndex1 += 16;
            }
        }

        bitonic_sort(A1in, A2in, B1in, B2in, C1in, C2in, D1in, D2in,
                     A1out, A2out, B1out, B2out, C1out, C2out, D1out, D2out);

        _mm512_store_si512(output + indexStart + 16 + (16 * i) + (mergeBlockSize * 0), A1out); //store in next memory spot
        _mm512_store_si512(output + indexStart + 16 + (16 * i) + (mergeBlockSize * 1), B1out); //store in next memory spot
        _mm512_store_si512(output + indexStart + 16 + (16 * i) + (mergeBlockSize * 2), C1out); //store in next memory spot
        _mm512_store_si512(output + indexStart + 16 + (16 * i) + (mergeBlockSize * 3), D1out); //store in next memory spot
    }

    //last 32
    A1in = A2out;
    B1in = B2out;
    C1in = C2out;
    D1in = D2out;
    A2in = (AIndex1 != currentBlockSize + (mergeBlockSize * 0)) ? _mm512_load_epi32(a + indexStart + AIndex1) : _mm512_load_epi32(a + indexStart + AIndex2);
    B2in = (BIndex1 != currentBlockSize + (mergeBlockSize * 1)) ? _mm512_load_epi32(a + indexStart + BIndex1) : _mm512_load_epi32(a + indexStart + BIndex2);
    C2in = (CIndex1 != currentBlockSize + (mergeBlockSize * 2)) ? _mm512_load_epi32(a + indexStart + CIndex1) : _mm512_load_epi32(a + indexStart + CIndex2);
    D2in = (DIndex1 != currentBlockSize + (mergeBlockSize * 3)) ? _mm512_load_epi32(a + indexStart + DIndex1) : _mm512_load_epi32(a + indexStart + DIndex2);
    bitonic_sort(A1in, A2in, B1in, B2in, C1in, C2in, D1in, D2in,
                 A1out, A2out, B1out, B2out, C1out, C2out, D1out, D2out);

    _mm512_store_si512(output + indexStart + (mergeBlockSize - 32) + (mergeBlockSize * 0), A1out);
    _mm512_store_si512(output + indexStart + (mergeBlockSize - 16) + (mergeBlockSize * 0), A2out);
    _mm512_store_si512(output + indexStart + (mergeBlockSize - 32) + (mergeBlockSize * 1), B1out);
    _mm512_store_si512(output + indexStart + (mergeBlockSize - 16) + (mergeBlockSize * 1), B2out);
    _mm512_store_si512(output + indexStart + (mergeBlockSize - 32) + (mergeBlockSize * 2), C1out);
    _mm512_store_si512(output + indexStart + (mergeBlockSize - 16) + (mergeBlockSize * 2), C2out);
    _mm512_store_si512(output + indexStart + (mergeBlockSize - 32) + (mergeBlockSize * 3), D1out);
    _mm512_store_si512(output + indexStart + (mergeBlockSize - 16) + (mergeBlockSize * 3), D2out);

    // delete temp_a;
}

int main(int argc, char *argv[])
{

    std::random_device rd; //Will be used to obtain a seed for the random number engine
    //std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<int> distrib(0, INT_MAX);
    //----------------------Initialize Array to be sorted----------------------
    int *a = (int *)aligned_alloc(64, sizeof(int) * ARRAY_LENGTH);
    for (int i = 0; i < ARRAY_LENGTH; i++) //Populates the array with random values
    {
        a[i] = distrib(gen);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(64)
    for (unsigned int i = 0; i < N; i += NUM_PER_THREAD)
    {
        int endIndex = i + NUM_PER_THREAD;
        //printf("Thread %d is ready to work within range [%d, %d).\n", omp_get_thread_num(), i, endIndex);
        //----------------------Sort into 16----------------------
        for (int j = i / 16; j < endIndex / 16; j++)
        {
            if (i == 0)
            {
                //printf("On first for loop\n");
            }
            sort(a + (j * 16), a + 16 + (j * 16)); //sort each 16
        }

        //----------------------Sort into 32----------------------
        for (int j = i; j < endIndex; j += 128)
        {
            if (i == 0)
            {
                //printf("On second for loop\n");
            }
            sort_32(a, j);
        }

        // ----------------------MERGE----------------------
        // int numMerges = (log(NUM_PER_THREAD) / log(2)) - 2;
        int numMerges = log(NUM_PER_SORTED) / log(2);
        int *threadInput = a + i;
        int *threadOutput = (int *)aligned_alloc(64, sizeof(int) * NUM_PER_THREAD);
        int *origOutput = threadOutput;
        for (int blockSize = 64; blockSize <= NUM_PER_SORTED; blockSize *= 2) //merge into larger sorted blocks
        {
            // cout << blockSize;
            //printf("blocksize %d\n", blockSize);
            for (int j = 0; j < NUM_PER_THREAD; j += (blockSize * 4))
            {
                if ((blockSize * 4) <= NUM_PER_THREAD)
                {
                    merge_blocks(threadInput, threadOutput, blockSize, j);
                }
            }
            int *temp = threadInput;
            threadInput = threadOutput;
            threadOutput = temp;
        }
        for (int j = 0; j < NUM_PER_THREAD; j++)
        {
            a[j + i] = threadInput[j];
        }
        delete origOutput;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    printf("It took %g milliseconds\n", fp_ms.count());
    // for (int i = 0; i < 10; i++)
    // {
    //     printf("%d ", a[i]);
    // }
    // printf("\n");

    // printArray(a, ARRAY_LENGTH, "output");

    // ----------------------Check if Sorted----------------------
    int numPerSorted = NUM_PER_SORTED;
    for (int j = 0; j < N; j += numPerSorted)
    {
        for (int k = j + 1; k < j + numPerSorted; k++)
        {
            if (a[k] < a[k - 1])
            {
                cout << "Not Sorted @ index: " << k << endl;
                cout << k - 1 << " = " << a[k - 1] << endl;
                cout << k << " = " << a[k] << endl;
                cout << k + 1 << " = " << a[k + 1] << endl;
                return 1;
            }
        }
        // printf("Block %d sorted\n", j);
    }

    // //----------------------debug----------------------
    // // printVectorInt(A1in, "A1in");
    // // printVectorInt(A2in, "A2in");
    // // printVectorInt(A1out, "A1out");
    // // printVectorInt(A2out, "A2out");
    // printArray(a, ARRAY_LENGTH, "output");
    // //----------------------debug----------------------

    cout << "SORTED" << endl;
    return 0;
}
