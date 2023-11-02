package MultiplyMethods;

public class SimpleMultiplication implements IMultiplyingMPI
{
    @Override
    public int[][] multiply(int[][] a, int[][] b) throws Exception {
        int rowsA = a.length;
        int colsA = a[0].length;
        int colsB = b[0].length;

        int[][] c = new int[rowsA][colsB];

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return c;
    }
    @Override
    public String toString()
    {
        return "\nSimple method";
    }
}
