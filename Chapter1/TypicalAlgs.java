public class TypicalAlgs
{
	// Euclid's Algorithm
	public static int gcd(int p, int q) {
		if (q == 0) return p;
		int r = p % q;
		return gcd(q, r);
	}

	public static boolean isPrime(int N) {
		if (N < 2) return false;
		for (int i = 2; i*i <= N; i++)
			if (N % i ==0)
				return false;
		return true;
	}
	
	// Newton's method
//	public static double sqrt(double c) {
//		if (c < 0) return Double.NaN;
//		double err = 1e-15;
//		double t = c;
//		while (Math.abs(t - c / t) > err * t)
//			t = (c/t + t) / 2.0;
//		return t;
//	}


	// https://en.wikipedia.org/wiki/Newton%27s_method
	// https://www.jianshu.com/p/dcd73888ac3a
	// x[n+1] = (x[n] + b / x[n]) / 2.0
	public static double NewtonSqrt(double c, double m) {
		double err = 1e-15;
		if (Math.abs(m * m - c) < err) return m;
		return NewtonSqrt(c, (m + c / m) / 2.0);
	}

	public static double sqrt(double c) {
		if (c < 0) return Double.NaN;
		return NewtonSqrt(c, c / 2.0);
	}

	// Binary Search method
	public static double DichotomySqrt(double c, double min, double max) {
		double err = 1e-15;
		if (Math.abs(max - min) < err) return (max + min) / 2.0;
		double t = (max + min) / 2.0;
		double s = t * t;
		if ( s > c) return DichotomySqrt(c, min, t);
		else if (s < c) return DichotomySqrt(c, t, max);
		else return t;
	}

//	public static double sqrt(double c) {
//		if (c < 0) return Double.NaN;
//		return DichotomySqrt(c, 0, c);
//	}

	public static void main(String [] args) {

		// test gcd
		if ( 4 == gcd(12, 16))
			System.out.println("Succeeded!");
		else
			System.out.println("Failed!");

		// test gcd
		if ( isPrime(13))
			System.out.println("Succeeded!");
		else
			System.out.println("Failed!");

		// test sqrt
		if ( 4 == sqrt(16))
			System.out.println("Succeeded!");
		else
			System.out.println("Failed!");

		System.out.println(sqrt(2));
	}
}