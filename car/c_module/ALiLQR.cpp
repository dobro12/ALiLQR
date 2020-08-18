#include <iostream>
#include <Python.h>
#include <math.h>
#include "mat.cpp"

#define TIME_HORIZON 20
#define X_DIM 3
#define U_DIM 2
#define COST_DIM 8

#define wheel_radius 0.05
#define wheel_pos_x -0.1
#define wheel_pos_y 0.13

#ifndef ERR
#define ERR 1
#endif
#define EPS 0.001

void transition(double* x, double* u, double* next_x, double delta_t);
void append(double* arr, int arr_pos, double* arr2, int len_arr2);
void get(double* arr, int arr_pos, int len, double* target);
void print(double*, int);
void print_idx(double* arr, int len, int mod, int idx);
mat get_A(double* x, double* u, double delta_t);
mat get_B(double* x, double* u, double delta_t);
mat get_cost(double* x, double* hazard_list, double hazard_radius);
mat get_C(double* x, double* hazard_list, double hazard_radius, double delta_t);
mat get_I(const mat & cost_vector, const mat & lambda_vector, const mat & mu_vector);

extern "C" {
    double* get_action(double* init_x, double delta_t, double damping_ratio, double learning_rate, double* R_data, double* Qf_data, 
                        double* Q_data, double* target_x, double* u_list, double* lambda_data, double* mu_data, double* hazard_list, double hazard_radius){
        mat R_mat(R_data, U_DIM, U_DIM);
        mat Q_mat(Q_data, X_DIM, X_DIM);
        mat Qf_mat(Qf_data, X_DIM, X_DIM);
        mat target_x_vector(target_x, X_DIM, 1);
        double x_list[X_DIM*(TIME_HORIZON+1)] = {};
        double new_x_list[X_DIM*(TIME_HORIZON+1)] = {};
        double new_u_list[U_DIM*TIME_HORIZON] = {};
        double J_value;
        mat x_vector, u_vector, temp_vector, temp_mat;
        double x[X_DIM], u[U_DIM], next_x[X_DIM];
        mat K_list[TIME_HORIZON], d_list[TIME_HORIZON];
        int max_cnt = 10; 
        int cnt;
        double pre_J_value;
        mat P_mat;
        mat p_vector;
        mat A_mat, B_mat;
        mat Qxx, Quu, Qux, Qxu, Qx, Qu, K_mat, d_vector;
        mat delta_x, delta_u;
        mat cost_vector, C_mat, I_mat;
        mat lambda_vector(lambda_data, COST_DIM, 1);
        mat mu_vector(mu_data, COST_DIM, 1);
        mat lambda_list[TIME_HORIZON + 1];
        double mu_scaling_factor = 2.0;

        J_value = 0;
        append(x_list, 0, init_x, X_DIM);
        for(int t_idx=0; t_idx<TIME_HORIZON; t_idx++){
            get(x_list, t_idx*X_DIM, X_DIM, x);
            get(u_list, t_idx*U_DIM, U_DIM, u);
            transition(x, u, next_x, delta_t);
            append(x_list, (t_idx+1)*X_DIM, next_x, X_DIM);

            x_vector = mat(x, X_DIM, 1);
            u_vector = mat(u, U_DIM, 1);
            temp_vector = x_vector - target_x_vector;
            temp_mat = temp_vector.transpose().matmul(Q_mat.matmul(temp_vector))*0.5;
            if(temp_mat.col != 1 || temp_mat.row != 1){
                std::cerr<<"[error!!] do not match dimension"<<std::endl;
                throw ERR;
            }
            J_value += temp_mat.data[0];
            temp_mat = u_vector.transpose().matmul(R_mat.matmul(u_vector))*0.5;
            if(temp_mat.col != 1 || temp_mat.row != 1){
                std::cerr<<"[error!!] do not match dimension"<<std::endl;
                throw ERR;
            }
            J_value += temp_mat.data[0];

            ///// initialize lambda list //////
            lambda_list[t_idx] = lambda_vector;
            ///////////////////////////////////
        }
        ///// initialize lambda list //////
        lambda_list[TIME_HORIZON] = lambda_vector;
        ///////////////////////////////////
        get(x_list, TIME_HORIZON*X_DIM, X_DIM, x);
        x_vector = mat(x, X_DIM, 1);
        temp_vector = x_vector - target_x_vector;
        temp_mat = temp_vector.transpose().matmul(Qf_mat.matmul(temp_vector))*0.5;
        if(temp_mat.col != 1 || temp_mat.row != 1){
            std::cerr<<"[error!!] do not match dimension"<<std::endl;
            throw ERR;
        }
        J_value += temp_mat.data[0];

        cnt = 0;
        while(max_cnt > cnt){
            pre_J_value = J_value;
            cnt++;

            /////// lambda and mu update ///////
            for(int t_idx=0; t_idx<TIME_HORIZON+1; t_idx++){
                get(x_list, t_idx*X_DIM, X_DIM, x);
                cost_vector = get_cost(x, hazard_list, hazard_radius);
                lambda_list[t_idx] = (lambda_list[t_idx] + mu_vector*cost_vector).max(0.0).min(10.0);
            }
            mu_vector = mu_vector*mu_scaling_factor;
            ////////////////////////////////////

            get(x_list, TIME_HORIZON*X_DIM, X_DIM, x);
            cost_vector = get_cost(x, hazard_list, hazard_radius);
            C_mat = get_C(x, hazard_list, hazard_radius, delta_t);
            I_mat = get_I(cost_vector, lambda_list[TIME_HORIZON], mu_vector);
            P_mat = Qf_mat + C_mat.transpose().matmul(I_mat.matmul(C_mat));
            x_vector = mat(x, X_DIM, 1);
            p_vector = Qf_mat.matmul(x_vector - target_x_vector) + C_mat.transpose().matmul(lambda_list[TIME_HORIZON] + I_mat.matmul(cost_vector));
            for(int t_idx=TIME_HORIZON-1; t_idx>=0; t_idx--){
                get(x_list, t_idx*X_DIM, X_DIM, x);
                get(u_list, t_idx*U_DIM, U_DIM, u);
                x_vector = mat(x, X_DIM, 1);
                u_vector = mat(u, U_DIM, 1);
                A_mat = get_A(x, u, delta_t);
                B_mat = get_B(x, u, delta_t);
                cost_vector = get_cost(x, hazard_list, hazard_radius);
                C_mat = get_C(x, hazard_list, hazard_radius, delta_t);
                I_mat = get_I(cost_vector, lambda_list[t_idx], mu_vector);
                Qxx = Q_mat + A_mat.transpose().matmul(P_mat.matmul(A_mat)) + C_mat.transpose().matmul(I_mat.matmul(C_mat));
                Quu = R_mat + B_mat.transpose().matmul(P_mat.matmul(B_mat));
                Qux = B_mat.transpose().matmul(P_mat.matmul(A_mat));
                Qxu = A_mat.transpose().matmul(P_mat.matmul(B_mat));
                Qx = Q_mat.matmul(x_vector - target_x_vector) + A_mat.transpose().matmul(p_vector) + C_mat.transpose().matmul(lambda_list[t_idx] + I_mat.matmul(cost_vector));
                Qu = R_mat.matmul(u_vector) + B_mat.transpose().matmul(p_vector);

                temp_mat = (Quu + mat(U_DIM)*damping_ratio).inverse()*(-1.0);
                K_mat = temp_mat.matmul(Qux);
                d_vector = temp_mat.matmul(Qu);

                P_mat = Qxx + K_mat.transpose().matmul(Quu.matmul(K_mat) + Qux) + Qxu.matmul(K_mat);
                p_vector = Qx + K_mat.transpose().matmul(Quu.matmul(d_vector) + Qu) + Qxu.matmul(d_vector);
                K_list[t_idx] = K_mat;
                d_list[t_idx] = d_vector;
            }

            append(new_x_list, 0, init_x, X_DIM);
            J_value = 0.0;
            for(int t_idx=0; t_idx<TIME_HORIZON; t_idx++){
                get(x_list, t_idx*X_DIM, X_DIM, x);
                x_vector = mat(x, X_DIM, 1);
                get(new_x_list, t_idx*X_DIM, X_DIM, x);
                temp_vector = mat(x, X_DIM, 1);
                delta_x = temp_vector - x_vector;
                get(u_list, t_idx*U_DIM, U_DIM, u);
                u_vector = mat(u, U_DIM, 1);
                delta_u = K_list[t_idx].matmul(delta_x) + d_list[t_idx]*learning_rate;
                append(new_u_list, t_idx*U_DIM, (u_vector + delta_u).data, U_DIM);
                get(new_u_list, t_idx*U_DIM, U_DIM, u);

                transition(x, u, next_x, delta_t);
                append(new_x_list, (t_idx+1)*X_DIM, next_x, X_DIM);

                x_vector = mat(x, X_DIM, 1);
                u_vector = mat(u, U_DIM, 1);
                temp_vector = x_vector - target_x_vector;
                temp_mat = temp_vector.transpose().matmul(Q_mat.matmul(temp_vector))*0.5;
                if(temp_mat.col != 1 || temp_mat.row != 1){
                    std::cerr<<"[error!!] do not match dimension"<<std::endl;
                    throw ERR;
                }
                J_value += temp_mat.data[0];
                temp_mat = u_vector.transpose().matmul(R_mat.matmul(u_vector))*0.5;
                if(temp_mat.col != 1 || temp_mat.row != 1){
                    std::cerr<<"[error!!] do not match dimension"<<std::endl;
                    throw ERR;
                }
                J_value += temp_mat.data[0];
            }
            get(new_x_list, TIME_HORIZON*X_DIM, X_DIM, x);
            x_vector = mat(x, X_DIM, 1);
            temp_vector = x_vector - target_x_vector;
            temp_mat = temp_vector.transpose().matmul(Qf_mat.matmul(temp_vector))*0.5;
            if(temp_mat.col != 1 || temp_mat.row != 1){
                std::cerr<<"[error!!] do not match dimension"<<std::endl;
                throw ERR;
            }
            J_value += temp_mat.data[0];

            memcpy(x_list, new_x_list, sizeof(double)*X_DIM*(TIME_HORIZON + 1));
            memcpy(u_list, new_u_list, sizeof(double)*U_DIM*TIME_HORIZON);
        }
        return u_list;
    }
}

void transition(double* x, double* u, double* next_x, double delta_t){
    double u1 = u[0], u2 = u[1];
    double pos_x = x[0], pos_y = x[1], theta = x[2];
    double omega = wheel_radius*(u2 - u1)/(2*wheel_pos_y);
    double vel_x = wheel_radius*u1*cos(theta) + omega*(wheel_pos_x*sin(theta) + wheel_pos_y*cos(theta));
    double vel_y = wheel_radius*u1*sin(theta) + omega*(wheel_pos_y*sin(theta) - wheel_pos_x*cos(theta));
    next_x[0] = pos_x + delta_t*vel_x;
    next_x[1] = pos_y + delta_t*vel_y;
    next_x[2] = theta + delta_t*omega;
    return;
}

void append(double* arr, int arr_pos, double* arr2, int len_arr2){
    for(int i=0;i<len_arr2;i++){
        arr[i+arr_pos] = arr2[i];
    }
}

void get(double* arr, int arr_pos, int len, double* target){
    for(int i=0;i<len;i++){
        target[i] = arr[arr_pos + i];
    }
}

mat get_A(double* x, double* u, double delta_t){
    double next_x[X_DIM];
    double temp_x[X_DIM];
    double temp_next_x[X_DIM];
    double eps = 0.001;
    double* data = new double[X_DIM*X_DIM];

    transition(x, u, next_x, delta_t);

    for(int i=0;i<X_DIM;i++){
        memcpy(temp_x, x, sizeof(double)*X_DIM);
        temp_x[i] += eps;
        transition(temp_x, u, temp_next_x, delta_t);
        for(int j=0;j<X_DIM;j++){
            data[j*X_DIM + i] = (temp_next_x[j] - next_x[j])/eps;
        }
    }

    mat A(data, X_DIM, X_DIM);
    delete[] data;
    return A;
}

mat get_B(double* x, double* u, double delta_t){
    double next_x[X_DIM];
    double temp_u[U_DIM];
    double temp_next_x[X_DIM];
    double eps = 0.001;
    double* data = new double[X_DIM*U_DIM];

    transition(x, u, next_x, delta_t);

    for(int i=0;i<U_DIM;i++){
        memcpy(temp_u, u, sizeof(double)*U_DIM);
        temp_u[i] += eps;
        transition(x, temp_u, temp_next_x, delta_t);
        for(int j=0;j<X_DIM;j++){
            data[j*U_DIM + i] = (temp_next_x[j] - next_x[j])/eps;
        }
    }

    mat B(data, X_DIM, U_DIM);
    delete[] data;
    return B;
}

mat get_cost(double* x, double* hazard_list, double hazard_radius){
    double hazard_pos[2];
    double cost[COST_DIM];
    for(int i=0; i<COST_DIM;i++){
        get(hazard_list, 2*i, 2, hazard_pos);
        cost[i] = hazard_radius - sqrt(pow(x[0] - hazard_pos[0], 2) + pow(x[1] - hazard_pos[1], 2));
    }
    return mat(cost, COST_DIM, 1);
}

mat get_C(double* x, double* hazard_list, double hazard_radius, double delta_t){
    double temp_x[X_DIM];
    double eps = 0.001;
    double* data = new double[COST_DIM*X_DIM];
    mat cost = get_cost(x, hazard_list, hazard_radius);
    mat temp_cost;

    for(int i=0;i<X_DIM;i++){
        memcpy(temp_x, x, sizeof(double)*X_DIM);
        temp_x[i] += eps;
        temp_cost = get_cost(temp_x, hazard_list, hazard_radius);
        for(int j=0;j<COST_DIM;j++){
            data[j*X_DIM + i] = (temp_cost.data[j] - cost.data[j])/eps;
        }
    }

    mat A(data, COST_DIM, X_DIM);
    delete[] data;
    return A;
}

mat get_I(const mat & cost_vector, const mat & lambda_vector, const mat & mu_vector){
    double* data = new double[COST_DIM*COST_DIM];
    for(int i=0;i<COST_DIM;i++){
        for(int j=0;j<COST_DIM;j++){
            if(i != j){
                data[i*COST_DIM + j] = 0;
                continue;
            }
            if(cost_vector.data[i] < 0)
                data[i*COST_DIM + j] = 0;
            else
                data[i*COST_DIM + j] = mu_vector.data[i];
        }
    }
    mat A(data, COST_DIM, COST_DIM);
    delete[] data;
    return A;
}

void print(double* arr, int len){
    std::cout<<"[";
    for(int i=0;i<len;i++){
        std::cout<<arr[i]<<", ";
    }
    std::cout<<"]"<<std::endl;
}

void print_idx(double* arr, int len, int mod, int idx){
    std::cout<<"[";
    for(int i=0;i<len;i++){
        if(i%mod == idx){
            std::cout<<arr[i]<<", ";
        }
    }
    std::cout<<"]"<<std::endl;
}
