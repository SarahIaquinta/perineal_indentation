#include "HyperElasAlgo.h"
#include "Objet_math.h"

HyperElasAlgo::HyperElasAlgo()
{
}


HyperElasAlgo::~HyperElasAlgo()
{
}

bool HyperElasAlgo::integrer_hgo3(int ndim, TENSEUR& epsi, VECTEUR& val_coef, MATRICE& dmatx, VECTEUR& sigma, double& Welas)
{
	double c1 = val_coef[0];
	double c2 = val_coef[1];
	double c3 = val_coef[2];
	double eps = val_coef[3];
	double k1 = val_coef[4];
	double k2 = val_coef[5];
	double teta = val_coef[6]; teta = teta * 3.1415926 / 180.0;

	if (ndim == 3) {
		int N = 1;
		MATRICE a(3, N);
		a[0][0] = cos(teta);
		a[1][0] = 0.0;
		a[2][0] = sin(teta);
		//      a[0][1]=cos(teta);
		//      a[1][1]=-sin(teta);
		//      a[2][1]=0.0;

		MATRICE E(3, 3), M(3, 3), Id(3, 3), C(3, 3), CM(3, 3), C2M(3, 3), CMMC(3, 3), MC(3, 3), S(3, 3), Cinv(3, 3), cofC(3, 3);

		E[0][0] = epsi[0];
		E[1][1] = epsi[1];
		E[2][2] = epsi[2];
		E[0][1] = E[1][0] = 0.5 * epsi[3];
		E[1][2] = E[2][1] = 0.5 * epsi[4];
		E[2][0] = E[0][2] = 0.5 * epsi[5];
		Id[0][0] = Id[1][1] = Id[2][2] = 1.0;
		C = 2.0 * E + Id;
		double J, I1, I2, I3, J4;
		Cinv = C.inverse(I3);
		J = sqrt(I3);
		cofC = I3 * Cinv.transpose();

		//invariants de C
		I1 = C[0][0] + C[1][1] + C[2][2];

	
		double cst, cst1, cst2, cst4, cst5, cst7, cst8, d1, d13, d3, d4, d11, d33, d31, d43, d44;
		int i, j, k, n, l;
		double A[3][3][3][3];
		for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) for (k = 0; k < 3; k++) for (l = 0; l < 3; l++)
			A[i][j][k][l] = 0.0;
		cst = pow(I3, -0.33333333333333333);
		cst1 = pow(I3, (-1.0 / 3.0));

		cst2 = pow(I3, (-2.0 / 3.0));
		cst4 = pow(I3, (-4.0 / 3.0));
		cst5 = pow(I3, (-5.0 / 3.0));
		cst7 = pow(I3, (-7.0 / 3.0));
		cst8 = pow(I3, (-8.0 / 3.0));


		d1 = c1 * cst1 + 2.0 * c2 * cst1 * (I1 * cst1 - 3.0) + 3.0 * c3 * cst1 * pow((I1 * cst1 - 3.0), 2.0);
		d3 = -(1.0 / 3.0) * c1 * I1 * cst4 - (2.0 / 3.0) * c2 * I1 * cst4 * (I1 * cst1 - 3.0) - c3 * I1 * cst4 * pow((I1 * cst1 - 3.0), 2.0) + eps * (1.0 - (1.0/sqrt(I3)));
		d11 = 2.0 * c2 * cst2 + 6.0 * c3 * cst2 * (I1 * cst1 - 3.0);	
		d31 = -(1.0 / 3.0) * c1 * cst4 - (2.0 / 3.0) * c2 * cst4 * (I1 * cst1 - 3.0) - (2.0 / 3.0) * c2 * I1 * cst5 - c3 * cst4 * pow((I1 * cst1 - 3.0), 2.0) - 2.0 * c3 * I1 * cst5 * (I1 * cst1 - 3.0);
		d33 = (4.0 / 9.0) * c1 * I1 * cst7 + (8.0 / 9.0) * c2 * I1 * cst7 * (I1 * cst1 - 3.0) + (2.0 / 9.0) * c2 * pow(I1, 2.0) * cst8 + (4.0 / 3.0) * c3 * I1 * cst7 * pow((I1 * cst1 - 3.0), 2.0) + (2.0 / 3.0) * c3 * pow(I1, 2.0) * cst8 * (I1 * cst1 - 3.0) + eps*(1.0 / (2.0*pow(I3 , (3.0 / 2.0))));


		for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) for (k = 0; k < 3; k++) for (l = 0; l < 3; l++) {

			A[i][j][k][l] = (
				d11 * Id[i][j] * Id[k][l] +
				d31 * (Id[i][j] * cofC[k][l] + cofC[i][j] * Id[k][l]) +
				d33 * cofC[i][j] * cofC[k][l] +
				d3 * I3 * (Cinv[i][j] * Cinv[k][l] - Cinv[i][k] * Cinv[j][l])
	
				);
		}
		S = d1  * Id  + d3 * cofC;
		for (n = 0; n < N; n++) {
			for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) M[i][j] = a[i][n] * a[j][n];
			CM = C * M;
			MC = M * C;
			CMMC = CM + MC;
			J4 = CM[0][0] + CM[1][1] + CM[2][2];
			double expo = exp(k2 * pow((J4 * cst1 - 1), 2.0));
			double dexpo = (J4 * cst1 - 1.0);
			if (J4 >= 1) {
				d4 = k1 * cst1 * dexpo * expo;
				d3 = -(1.0 / 3.0) * k1 * J4 * cst4 * dexpo * expo;
				d43 = -(1.0 / 3.0) * k1 * cst4 * dexpo * expo - (1.0 / 3.0) * k1 * J4 * cst5 * expo - (2.0 / 3.0) * J4 * k2 * k1 * cst5 * pow(dexpo, 2.0) * expo;
				d44 = k1 * cst2 * expo + 2.0 * k1 * k2 * cst2 * pow(dexpo, 2.0) * expo;
				d33 = (4.0 / 9.0) * k1 * J4 * cst7 * dexpo * expo + (1.0 / 9.0) * k1 * pow(J4, 2.0) * cst8 * expo + (2.0 / 9.0) * k1 * k2 * pow(J4, 2.0) * cst4 * pow(dexpo, 2.0) * expo;

			}
			else {
				d4 =0.0; 

				d3 =0.0; 

				d43=0.0; 

				d44=0.0; 

				d33 = 0.0;


			}
			//densite d energie de deformation
			double Welas = 0.0;
			//Seconde tenseur de contraintes de Piola-Kirshroff
			S = S + d4 * M + d3 * cofC;
			//c*** matrice D

			for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) for (k = 0; k < 3; k++) for (l = 0; l < 3; l++) {

				A[i][j][k][l] += (
					d43 * (cofC[i][j] * M[k][l] + M[i][j] * cofC[k][l]) +
					d44 * M[i][j] * M[k][l] +
					d33 * cofC[i][j] * cofC[k][l] +
					d3 * I3 * (Cinv[i][j] * Cinv[k][l] - Cinv[i][k] * Cinv[j][l])
					);
			}
		}

		sigma[0] = 2.0 * S[0][0];
		sigma[1] = 2.0 * S[1][1];
		sigma[2] = 2.0 * S[2][2];
		sigma[3] = 2.0 * S[0][1];
		sigma[4] = 2.0 * S[1][2];
		sigma[5] = 2.0 * S[2][0];

		dmatx[0][0] = 4.0 * A[0][0][0][0];
		dmatx[1][1] = 4.0 * A[1][1][1][1];
		dmatx[2][2] = 4.0 * A[2][2][2][2];
		dmatx[3][3] = 2.0 * (A[0][1][0][1] + A[0][1][1][0]);
		dmatx[4][4] = 2.0 * (A[1][2][1][2] + A[1][2][2][1]);
		dmatx[5][5] = 2.0 * (A[0][2][0][2] + A[0][2][2][0]);

		dmatx[0][1] = dmatx[1][0] = 4.0 * A[0][0][1][1];
		dmatx[0][2] = dmatx[2][0] = 4.0 * A[0][0][2][2];
		dmatx[0][3] = dmatx[3][0] = 4.0 * A[0][0][0][1];
		dmatx[0][4] = dmatx[4][0] = 4.0 * A[0][0][1][2];
		dmatx[0][5] = dmatx[5][0] = 4.0 * A[0][0][0][2];

		dmatx[1][2] = dmatx[2][1] = 4.0 * A[1][1][2][2];
		dmatx[1][3] = dmatx[3][1] = 4.0 * A[1][1][0][1];
		dmatx[1][4] = dmatx[4][1] = 4.0 * A[1][1][1][2];
		dmatx[1][5] = dmatx[5][1] = 4.0 * A[1][1][0][2];

		dmatx[2][3] = dmatx[3][2] = 4.0 * A[2][2][0][1];
		dmatx[2][4] = dmatx[4][2] = 4.0 * A[2][2][1][2];
		dmatx[2][5] = dmatx[5][2] = 4.0 * A[2][2][0][2];

		dmatx[3][4] = dmatx[4][3] = 2.0 * (A[0][1][1][2] + A[0][1][2][1]);
		dmatx[3][5] = dmatx[5][3] = 2.0 * (A[0][1][0][2] + A[0][1][2][0]);

		dmatx[4][5] = dmatx[5][4] = 2.0 * (A[0][2][1][2] + A[0][2][2][1]);
	}
	else if (ndim == 2) {
	
	}
	else return false;

	return true;

}

bool HyperElasAlgo::integrer_ArrudaBoyce(int ndim, TENSEUR& epsi, VECTEUR& val_coef, MATRICE& dmatx, VECTEUR& sigma, double& Welas)
{
	
	return false;
}