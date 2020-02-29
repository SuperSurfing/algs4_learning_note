public class BinarySearch {
    public static int rank(int key, int[] a) {
        return rank(key, a, 0, a.length - 1);
    }

    public static int rank(int key, int[] a, int lo, int hi) {
        if (lo > hi) return -1;
        int mi = lo + (hi - lo) / 2;
        if (key < a[mi]) return rank(key, a, lo, mi);
        else if (key > a[mi]) return rank(key, a, mi, hi);
        else return mi;
    }

    public static void main(String[] args) {
        // test
        int[] array = {0, 1, 2, 3, 4, 5};
        if ( 3 == rank(3, array))
            System.out.println("Succeeded!");
        else
            System.out.println("Failed!");

        // test
        int[] array2 = {5, 8, 4, 9, 7, 2};
        if ( 0 == rank(5, array2))
            System.out.println("Succeeded!");
        else
            System.out.println("Failed!");
    }
}
