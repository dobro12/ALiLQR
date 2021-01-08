#ifndef MAT
#define MAT

#include <iostream>
#include <math.h>

#ifndef ERR
#define ERR 1
#endif

class mat{
    public:
    double* data;
    int row, col;

    mat(){
        row = 2;
        col = 2;
        data = new double[row*col];
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                data[i*col + j] = 0.0;
            }
        }
    }
    mat(int n){
        row = n;
        col = n;
        data = new double[row*col];
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                data[i*col + j] = 0.0;
            }
        }
    }
    mat(int r, int c){
        row = r;
        col = c;
        data = new double[row*col];
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                data[i*col + j] = 0.0;
            }
        }
    }
    mat(double* ref, int r, int c){
        row = r;
        col = c;
        data = new double[row*col];
        memcpy(data, ref, sizeof(double)*row*col);
    }
    mat(const mat& A){
        //std::cout<<"copy destructor"<<std::endl;
        row = A.row;
        col = A.col;
        data = new double[row*col];
        memcpy(data, A.data, sizeof(double)*row*col);
    }
    ~mat(){
        //std::cout<<"destructor"<<std::endl;
        delete [] data;
    }

    mat & operator =(const mat & A){
        //std::cout<<"copy operator"<<std::endl;
        row = A.row;
        col = A.col;
        delete [] data;
        data = new double[row*col];
        memcpy(data, A.data, sizeof(double)*row*col);
        return *this;
    }
    const mat operator +(const mat & A) const{
        if(col != A.col || row != A.row){
            std::cerr<<"[error!!] do not match dimension"<<std::endl;
            throw ERR;
        }
        double* temp_data = new double[row*col];
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                temp_data[i*col + j] = data[i*col + j] + A.data[i*col + j];
            }
        }
        mat B(temp_data, row, col);
        delete[] temp_data;
        return B;        
    }
    const mat operator -(const mat & A) const{
        if(col != A.col || row != A.row){
            std::cerr<<"[error!!] do not match dimension"<<std::endl;
            throw ERR;
        }
        double* temp_data = new double[row*col];
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                temp_data[i*col + j] = data[i*col + j] - A.data[i*col + j];
            }
        }
        mat B(temp_data, row, col);
        delete[] temp_data;
        return B;        
    }
    const mat operator *(const double & A) const{
        double* temp_data = new double[row*col];
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                temp_data[i*col + j] = data[i*col + j] * A;
            }
        }
        mat B(temp_data, row, col);
        delete[] temp_data;
        return B;        
    }
    const mat operator *(const mat & A) const{
        if(col != A.col || row != A.row){
            std::cerr<<"[error!!] do not match dimension"<<std::endl;
            throw ERR;
        }

        double* temp_data = new double[row*col];
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                temp_data[i*col + j] = data[i*col + j] * A.data[i*col + j];
            }
        }
        mat B(temp_data, row, col);
        delete[] temp_data;
        return B;        
    }

    const mat max(const double & a) const{
        double* temp_data = new double[row*col];
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                if(data[i*col + j] > a)
                    temp_data[i*col + j] = data[i*col + j];
                else
                    temp_data[i*col + j] = a;
            }
        }
        mat A(temp_data, col, row);
        delete[] temp_data;
        return A;
    }

    const mat min(const double & a) const{
        double* temp_data = new double[row*col];
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                if(data[i*col + j] < a)
                    temp_data[i*col + j] = data[i*col + j];
                else
                    temp_data[i*col + j] = a;
            }
        }
        mat A(temp_data, col, row);
        delete[] temp_data;
        return A;
    }

    const mat transpose() const{
        double* temp_data = new double[row*col];
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                temp_data[j*row + i] = data[i*col + j];
            }
        }
        mat A(temp_data, col, row);
        delete[] temp_data;
        return A;
    }

    void print() const{
        std::cout<<"[";
        for(int i=0;i<row;i++){
            std::cout<<"[";
            for(int j=0;j<col;j++){
                std::cout<<data[i*col + j]<<", ";
            }
            if(i == row-1)
                std::cout<<"]";
            else
                std::cout<<"]"<<std::endl;
        }
        std::cout<<"]"<<std::endl;
    }

    double magnitude() const{
        double mag = 0;
        for(int i=0;i<row;i++){
            for(int j=0;j<col;j++){
                mag += data[i*col + j]*data[i*col + j];
            }
        }
        mag = sqrt(mag);
        return mag;
    }

    double dot(const mat& A) const{
        if(!(col == 1 && A.col == 1 && row == A.row)){
            std::cerr<<"[error!!] do not match dimension"<<std::endl;
            throw ERR;
        }
        double mag = 0;
        for(int i=0;i<row;i++){
            mag += data[i]*A.data[i];
        }
        return mag;
    }

    const mat matmul(const mat& A) const{
        mat result(row, A.col);
        if(col != A.row){
            std::cerr<<"[error!!] do not match dimension"<<std::endl;
            throw ERR;
        }
        for(int i=0;i<row;i++){
            for(int j=0;j<A.col;j++){
                double value = 0.0;
                for(int k=0;k<col;k++){
                    value += data[i*col + k]*A.data[k*A.col + j];
                }
                result.data[i*A.col + j] = value;
            }
        }
        return result;
    }

    const mat getCofactor(int p, int q) const{ 
        int idx = 0; 
        mat A(row-1, col-1);
    
        for (int i=0; i<row; i++){ 
            for (int j=0; j<col; j++){
                if (i != p && j != q){
                    A.data[idx] = data[i*col + j]; 
                    idx++;
                } 
            } 
        }
        return A; 
    } 

    double determinant() const{
        double D = 0;
        double sign = 1.0;

        if(col != row){
            std::cerr<<"[error!!] do not match dimension"<<std::endl;
            throw ERR;
        }

        if(row == 1)
            return data[0];
        else if(row == 2)
            return data[0]*data[3] - data[1]*data[2];

        for(int i=0;i<row;i++){
            D += sign*data[i]*this->getCofactor(0, i).determinant();
            sign = -sign;
        }
        return D;
    }

    const mat adjoint() const{ 
        mat result(row, col);
        double sign;

        if(col != row){
            std::cerr<<"[error!!] do not match dimension"<<std::endl;
            throw ERR;
        }

        if (row == 1) 
        { 
            result.data[0] = 1.0; 
            return result;
        } 
    
        for (int i=0; i<row; i++){ 
            for (int j=0; j<col; j++){
                sign = ((i+j)%2==0)? 1: -1; 
                result.data[j*row + i] = this->getCofactor(i, j).determinant()*sign;
            } 
        }
        return result; 
    } 

    const mat inverse() const{ 
        double det;
        mat result(row, col);
        mat adjoint;

        if(col != row){
            std::cerr<<"[error!!] do not match dimension"<<std::endl;
            throw ERR;
        }

        det = this->determinant(); 
        if (det == 0){ 
            std::cerr<<"[error!!] determinant is zero."<<std::endl;
            throw ERR;
        } 
    
        adjoint = this->adjoint(); 
        for (int i=0; i<row; i++) 
            for (int j=0; j<col; j++) 
                result.data[i*col + j] = adjoint.data[i*col + j]/det; 
        return result; 
    }

    const mat inverse_matmul_column_vector(const mat& g, int num_conjugate=10) const{
        if(!(g.col == 1 && row == g.row && row == col)){
            std::cerr<<"[error!!] do not match dimension"<<std::endl;
            throw ERR;
        }

        mat x_value(g.row, g.col);
        mat Ap;
        mat residue = g;
        mat p_vector = g;
        double rs_old = residue.dot(residue);
        double rs_new;
        double pAp;
        double alpha;

        for(int i=0;i<num_conjugate;i++){
            Ap = this->matmul(p_vector);
            pAp = p_vector.dot(Ap);
            alpha = rs_old/pAp;
            x_value = x_value + p_vector*alpha;
            residue = residue - Ap*alpha;
            rs_new = residue.dot(residue);
            if(sqrt(rs_new) < 1e-5)
                break;
            p_vector = residue + p_vector*(rs_new/(1e-10 + rs_old));
            rs_old = rs_new;
        }

        return x_value;
    }

    const mat inverse_matmul(const mat& g, int num_conjugate=10) const{
        if(!(row == g.row && row == col)){
            std::cerr<<"[error!!] do not match dimension"<<std::endl;
            throw ERR;
        }
        if(g.col == 1){
            return this->inverse_matmul_column_vector(g, num_conjugate);
        }

        mat transpose_x_value(g.col, col);
        mat g_transpose = g.transpose();
        for(int i=0;i<g.col;i++){
            mat temp_x_value(&g_transpose.data[i*col], col, 1);
            temp_x_value = this->inverse_matmul_column_vector(temp_x_value, num_conjugate);
            memcpy(&transpose_x_value.data[i*col], temp_x_value.data, sizeof(double)*col);
        }
        return transpose_x_value.transpose();

    }

    const mat cross(const mat& A) const{
        mat result(row, col);
        if(!(col == 1 && A.col == 1 && row == 3 && A.row == 3)){
            std::cerr<<"[error!!] do not match dimension"<<std::endl;
            throw ERR;
        }
        result.data[0] = data[1]*A.data[2] - data[2]*A.data[1];
        result.data[1] = data[2]*A.data[0] - data[0]*A.data[2];
        result.data[2] = data[0]*A.data[1] - data[1]*A.data[0];
        return result;
    }

    static const mat x_rot(const double yaw) {
        double temp_data[] = {1.0, 0.0, 0.0,
                            0.0, cos(yaw), -sin(yaw),
                            0.0, sin(yaw), cos(yaw)};
        mat rot(temp_data, 3, 3);
        return rot;
    }

    static const mat y_rot(const double yaw) {
        double temp_data[] = {cos(yaw), 0.0, sin(yaw),
                            0.0, 1.0, 0.0,
                            -sin(yaw), 0.0, cos(yaw)};
        mat rot(temp_data, 3, 3);
        return rot;
    }

    static const mat z_rot(const double yaw) {
        double temp_data[] = {cos(yaw), -sin(yaw), 0.0,
                            sin(yaw), cos(yaw), 0.0,
                            0.0, 0.0, 1.0};
        mat rot(temp_data, 3, 3);
        return rot;
    }

    static const mat rpy_rot(double* rpy){
        mat temp_mat = x_rot(rpy[0]);
        temp_mat = y_rot(rpy[1]).matmul(temp_mat);
        temp_mat = z_rot(rpy[2]).matmul(temp_mat);
        return temp_mat;
    }

    static const mat eye(int dim){
        double* temp_data = new double[dim*dim];
        for(int i=0;i<dim;i++){
            for(int j=0;j<dim;j++){
                if(i == j)
                    temp_data[i*dim + j] = 1.0;
                else
                    temp_data[i*dim + j] = 0.0;
            }
        }
        mat A(temp_data, dim, dim);
        delete[] temp_data;
        return A;
    }
};

#endif