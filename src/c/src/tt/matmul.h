#pragma once


// C += alpha * (A @ B)
void matmul(void* A, void* B, void* C, float alpha, bool transpose_A=false, bool transpose_B=false);

