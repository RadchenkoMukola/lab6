package MultiplyMethods;
import mpi.MPI;

public class CannonMultiplyingMPI implements IMultiplyingMPI {
    @Override
    public String toString() {
        return "\nCannon method";
    }

    @Override
    public int[][] multiply(int[][] A, int[][] B) throws Exception {
        // Отримуємо ранг та кількість процесів у MPI комунікації
        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();

        // Алгоритм множення матриці Cannon
        int n = (int) Math.sqrt(size);
        int row = rank / n;
        int col = rank % n;
        int matrixSize = A.length;

        // Перевірка на коректність параметрів
        if (n * n != size || matrixSize % n != 0) {
            if (rank == 0)
                throw new Exception("!!!Number of processes must be a perfect square!!!");
            else
                return null;
        }

        int subMatrixSize = A[0].length;
        int blockSize = matrixSize / n;
        int[][] C = new int[subMatrixSize][subMatrixSize];

        // Ініціалізація початкових даних для обчислень
        ShiftLeft(A, row, subMatrixSize, matrixSize);
        ShiftUp(B, col, subMatrixSize, matrixSize);

        for (int k = 0; k < blockSize; k++) {
            // Локальне множення
            for (int i = 0; i < subMatrixSize; i++) {
                for (int j = 0; j < subMatrixSize; j++) {
                    for (int m = 0; m < subMatrixSize; m++)
                        C[i][j] += A[i][m] * B[m][j];
                }
            }
            // Ротація матриці A вліво та матриці B вгору
            ShiftLeft(A, 1, subMatrixSize, matrixSize);
            ShiftUp(B, 1, subMatrixSize, matrixSize);
        }

        return C;
    }

    public void ShiftLeft(int[][] matrix, int shift, int subMatrixSize, int matrixSize) throws Exception {
        int rank = MPI.COMM_WORLD.Rank();
        int row = rank / matrixSize;
        int prevCol = (rank - shift + matrixSize) % matrixSize;
        int nextCol = (rank + shift) % matrixSize;
        int sourceRank = row * matrixSize + prevCol;
        int destRank = row * matrixSize + nextCol;
        int[] buffer = new int[subMatrixSize * subMatrixSize];

        // Плоский масив для відправки
        for (int i = 0; i < subMatrixSize; i++) {
            for (int j = 0; j < subMatrixSize; j++)
                buffer[i * subMatrixSize + j] = matrix[i][j];
        }

        int[] recvBuffer = new int[subMatrixSize * subMatrixSize];
        MPI.COMM_WORLD.Sendrecv(buffer, 0, buffer.length, MPI.INT, destRank, 0, recvBuffer, 0, recvBuffer.length, MPI.INT, sourceRank, 0);

        // Розгортання отриманих даних в матрицю
        for (int i = 0; i < subMatrixSize; i++) {
            for (int j = 0; j < subMatrixSize; j++)
                matrix[i][j] = recvBuffer[i * subMatrixSize + j];
        }
    }

    public void ShiftUp(int[][] matrix, int shift, int subMatrixSize, int matrixSize) throws Exception {
        int rank = MPI.COMM_WORLD.Rank();
        int col = rank % matrixSize;
        int prevRow = (rank - shift * matrixSize + matrixSize * matrixSize) % (matrixSize * matrixSize) / matrixSize;
        int nextRow = (rank + shift * matrixSize) % (matrixSize * matrixSize) / matrixSize;
        int sourceRank = prevRow * matrixSize + col;
        int destRank = nextRow * matrixSize + col;
        int[] buffer = new int[subMatrixSize * subMatrixSize];

        // Плоский масив для відправки
        for (int i = 0; i < subMatrixSize; i++) {
            for (int j = 0; j < subMatrixSize; j++)
                buffer[i * subMatrixSize + j] = matrix[i][j];
        }

        int[] recvBuffer = new int[subMatrixSize * subMatrixSize];
        MPI.COMM_WORLD.Sendrecv(buffer, 0, buffer.length, MPI.INT, destRank, 0, recvBuffer, 0, recvBuffer.length, MPI.INT, sourceRank, 0);

        // Розгортання отриманих даних в матрицю
        for (int i = 0; i < subMatrixSize; i++) {
            for (int j = 0; j < subMatrixSize; j++)
                matrix[i][j] = recvBuffer[i * subMatrixSize + j];
        }
    }
}