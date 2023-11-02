package MultiplyMethods;
import mpi.MPI;

public class FoxMultiplyingMPI implements IMultiplyingMPI {

    // Метод для множення матриць за допомогою алгоритму Fox у контексті MPI
    public int[][] multiply(int[][] A, int[][] B) throws Exception {
        // Отримання рангу та розміру комунікації MPI
        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();
        // Визначення розмірності сітки
        int n = (int) Math.sqrt(size);
        // Визначення розміру матриць
        int matrixSize = A.length;
        // Перевірка, чи кількість процесів є квадратом та чи матриця може бути рівномірно розділена
        if (n * n != size || matrixSize % n != 0) {
            if (rank == 0)
                throw new Exception("!!!Number of processes must be a perfect square!!!");
            else
                return null;
        }
        // Ініціалізація локального C
        int[][] localC = new int[matrixSize][A[0].length];
        // Обчислення рядка і стовпця поточного процесу в сітці
        int i = rank / n;
        int j = rank % n;
        // Ініціалізація буферів для передачі та отримання даних
        Object[] sendBuffer = new Object[1];
        Object[] recvBuffer = new Object[1];

        // Цикл для кожного кроку алгоритму Fox
        for (int l = 0; l < n - (matrixSize - 1); l++) {
            // Очищення буферів
            sendBuffer[0] = null;
            recvBuffer[0] = null;
            // Синхронізація всіх процесів
            MPI.COMM_WORLD.Barrier();
            // Визначення поточного кореня
            int root = i * n + (j + l) % n;

            // Цикл для розсилки та отримання підматриць A між процесами
            for (int broadcaster = 0; broadcaster < size; broadcaster++) {
                if (rank == broadcaster) {
                    // Передача підматриці A іншим процесам
                    sendBuffer[0] = A;
                    for (int proc = 0; proc < size; proc++) {
                        if (proc != broadcaster)
                            MPI.COMM_WORLD.Send(sendBuffer, 0, 1, MPI.OBJECT, proc, 99);
                    }
                } else {
                    // Отримання підматриці A від іншого процесу
                    MPI.COMM_WORLD.Recv(recvBuffer, 0, 1, MPI.OBJECT, broadcaster, 99);
                    if (broadcaster == root)
                        A = (int[][]) recvBuffer[0];
                }
            }
            // Виклик функції для множення підматриць A і B
            matrixMultiply(A, B, localC, matrixSize);
            // Визначення номерів сусідніх процесів
            int leftNeighbor = i * n + (j - 1 + n) % n;
            int rightNeighbor = i * n + (j + 1) % n;
            // Підготовка та виконання передачі та отримання підматриці B від сусідніх процесів
            sendBuffer[0] = B;
            MPI.COMM_WORLD.Sendrecv(sendBuffer, 0, 1, MPI.OBJECT, leftNeighbor, 0, recvBuffer, 0, 1, MPI.OBJECT, rightNeighbor, 0);
            B = (int[][]) recvBuffer[0];
        }
        // Повернення результату
        return localC;
    }

    // Функція для множення двох матриць
    public void matrixMultiply(int[][] A, int[][] B, int[][] C, int blockSize) {
        // Цикл для обчислення кожного елементу матриці C
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < blockSize; j++) {
                for (int k = 0; k < blockSize; k++)
                    C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    @Override
    public String toString() {
        return "\nFox method";
    }
}