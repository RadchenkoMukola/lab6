package MultiplyMethods;
import mpi.*;

public class TapeMultiplyingMPI implements IMultiplyingMPI {

    // Метод для множення матриць за допомогою алгоритму "стрижневого" розподілення даних у контексті MPI
    public int[][] multiply(int[][] a, int[][] b) throws Exception {
        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();
        int rowsPerProcess = a.length / size;
        int remainder = a.length % size;
        int[] sendcounts = new int[size];
        int[] displs = new int[size];
        int count = 0;

        // Обчислення кількості рядків для кожного процесу та визначення розподілу даних
        for (int i = 0; i < size; i++) {
            if (i < remainder)
                sendcounts[i] = rowsPerProcess + 1;
            else
                sendcounts[i] = rowsPerProcess;
            displs[i] = count;
            count += sendcounts[i];
        }

        int[][] localA = new int[sendcounts[rank]][a[0].length];

        if (rank == 0) {
            // На процесі з рангом 0 відправляємо дані іншим процесам
            for (int i = 1; i < size; i++)
                MPI.COMM_WORLD.Send(a, displs[i], sendcounts[i], MPI.OBJECT, i, 0);
            // Процес 0 отримує свою частину даних
            System.arraycopy(a, 0, localA, 0, sendcounts[0]);
        } else {
            // Інші процеси отримують свою частину даних
            MPI.COMM_WORLD.Recv(localA, 0, sendcounts[rank], MPI.OBJECT, 0, 0);
        }

        // Множення локального сегмента матриці
        int[][] localResult = new int[localA.length][b[0].length];
        for (int row = 0; row < localA.length; row++) {
            for (int col = 0; col < b[0].length; col++) {
                for (int k = 0; k < b.length; k++)
                    localResult[row][col] += localA[row][k] * b[k][col];
            }
        }

        int[][] res = null;
        if (rank == 0) {
            // Процес з рангом 0 отримує результат та складає його разом
            res = new int[a.length][b[0].length];
            System.arraycopy(localResult, 0, res, 0, sendcounts[0]);
            for (int i = 1; i < size; i++)
                MPI.COMM_WORLD.Recv(res, displs[i], sendcounts[i], MPI.OBJECT, i, 1);
        } else {
            // Інші процеси відправляють свої результати
            MPI.COMM_WORLD.Send(localResult, 0, localResult.length, MPI.OBJECT, 0, 1);
        }
        return res;
    }

    @Override
    public String toString() {
        return "\nTape method";
    }
}