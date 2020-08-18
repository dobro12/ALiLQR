#include <iostream>
#include <Python.h>
#include <math.h>
#include "mat.cpp"

#define X_DIM 12
#define U_DIM 12
#define CONST_DIM 17
#define NUM_LEG 4

#ifndef ERR
#define ERR 1
#endif
#define EPS 0.001

double leg_mass = 0.54 + 0.634 + 0.064 + 0.15;
double mass = 1.0*(3.3 + 4*leg_mass);
double graivty_data[] = {0, 0, -9.8};
double base_com_pos_data[] = {1.0e-3, 0.0, 0.0};
double inertia_data[] = {0.011253, 0, 0, 
                        0, 0.036203, 0, 
                        0, 0, 0.042673};
double abduct_org_data[] =  {0.19, -0.049, 0.0, 
                            0.19, 0.049, 0.0,
                            -0.19, -0.049, 0.0, 
                            -0.19, 0.049, 0.0};
mat gravity(graivty_data, 3, 1);
mat base_com_pos(base_com_pos_data, 3, 1);
mat inertia(inertia_data, 3, 3);
mat abduct_org_list[] = {mat(&abduct_org_data[0], 3, 1), 
                        mat(&abduct_org_data[3], 3, 1),
                        mat(&abduct_org_data[6], 3, 1),
                        mat(&abduct_org_data[9], 3, 1)};

mat transition(const mat & x, const mat & u, const mat & foot_pos_list, double delta_t);
mat get_const(const mat & x, const mat & u, const mat & foot_pos_data, const mat & contact_phi_list);
mat get_A(const mat & x, const mat & u, const mat & foot_pos_list, double delta_t);
mat get_B(const mat & x, const mat & u, const mat & foot_pos_list, double delta_t);
mat get_Cx(const mat & x, const mat & u, const mat & foot_pos_list, const mat & contact_phi_list);
mat get_Cu(const mat & x, const mat & u, const mat & foot_pos_list, const mat & contact_phi_list);
mat get_I(const mat & const_vector, const mat & lambda_vector, const mat & mu_vector);

extern "C" {
    void get_action(int time_horizon, double* init_x_data, double* init_u_data, double* delta_time_list, double* foot_pos_data, double* contact_phi_data, double* target_x_data, double* target_u_data, double damping_ratio, double learning_rate, int max_iteration, double* R_data, double* Qf_data, double* Q_data, double* init_lambda_data, double* init_mu_data, double mu_scaling_factor, double max_lambda, double* return_var){
        // ######## declare variables ########
        mat Qf_mat(Qf_data, X_DIM, X_DIM);
        mat Q_mat(Q_data, X_DIM, X_DIM);
        mat R_mat(R_data, U_DIM, U_DIM);

        mat init_lambda_vector(init_lambda_data, CONST_DIM, 1);
        mat init_mu_vector(init_mu_data, CONST_DIM, 1);
        mat init_x(init_x_data, X_DIM, 1);

        mat* foot_pos_list = new mat[time_horizon + 1];
        mat* contact_phi_list = new mat[time_horizon + 1];

        mat* x_list = new mat[time_horizon + 1];
        mat* u_list = new mat[time_horizon];
        mat* new_x_list = new mat[time_horizon + 1];
        mat* new_u_list = new mat[time_horizon];
        mat* target_x_list = new mat[time_horizon + 1];
        mat* target_u_list = new mat[time_horizon];

        mat* K_list = new mat[time_horizon];
        mat* d_list = new mat[time_horizon];

        mat* lambda_list = new mat[time_horizon + 1];

        double J_value;
        double pre_J_value;
        double delta_time;
        double temp_learning_rate;
        int cnt;

        mat lambda_vector(CONST_DIM, 1);
        mat mu_vector(CONST_DIM, 1);
        mat x(X_DIM, 1);
        mat next_x(X_DIM, 1);
        mat u(U_DIM, 1);
        mat temp_vector, temp_mat;
        mat P_mat(X_DIM, X_DIM);
        mat p_vector(X_DIM, 1);
        mat A_mat;
        mat B_mat;
        mat C_x_mat;
        mat C_u_mat;
        mat Qxx, Quu, Qux, Qxu, Qx, Qu, K_mat, d_vector;
        mat delta_x, delta_u;
        mat const_vector;
        mat I_mat;
        // ###################################

        // ###### initilze variables ######
        for(int i=0;i<time_horizon+1;i++){
            if(i < time_horizon){
                u_list[i] = mat(&init_u_data[i*U_DIM], U_DIM, 1);
                target_u_list[i] = mat(&target_u_data[i*U_DIM], U_DIM, 1);
            }
            target_x_list[i] = mat(&target_x_data[i*X_DIM], X_DIM, 1);
            foot_pos_list[i] = mat(&foot_pos_data[i*NUM_LEG*3], NUM_LEG*3, 1);
            contact_phi_list[i] = mat(&contact_phi_data[i*NUM_LEG], NUM_LEG, 1);
            lambda_list[i] = init_lambda_vector;
        }
        // ################################

        // ###### get x list ######
        J_value = 0;
        x_list[0] = init_x;
        mu_vector = init_mu_vector;
        for(int t_idx=0; t_idx<time_horizon; t_idx++){
            x = x_list[t_idx];
            u = u_list[t_idx];
            delta_time = delta_time_list[t_idx];
            next_x = transition(x, u, foot_pos_list[t_idx], delta_time);
            x_list[t_idx + 1] = next_x;

            temp_vector = x - target_x_list[t_idx];
            temp_mat = temp_vector.transpose().matmul(Q_mat.matmul(temp_vector))*0.5;
            if(temp_mat.col != 1 || temp_mat.row != 1){
                std::cerr<<"[error!!] do not match dimension"<<std::endl;
                throw ERR;
            }
            J_value += temp_mat.data[0];
            temp_vector = u - target_u_list[t_idx];
            temp_mat = temp_vector.transpose().matmul(R_mat.matmul(temp_vector))*0.5;
            if(temp_mat.col != 1 || temp_mat.row != 1){
                std::cerr<<"[error!!] do not match dimension"<<std::endl;
                throw ERR;
            }
            J_value += temp_mat.data[0];
        }
        x = x_list[time_horizon];
        temp_vector = x - target_x_list[time_horizon];
        temp_mat = temp_vector.transpose().matmul(Qf_mat.matmul(temp_vector))*0.5;
        if(temp_mat.col != 1 || temp_mat.row != 1){
            std::cerr<<"[error!!] do not match dimension"<<std::endl;
            throw ERR;
        }
        J_value += temp_mat.data[0];
        // ########################

        cnt = 0;
        while(cnt < max_iteration){
            pre_J_value = J_value;
            cnt++;

            // ######## lambda and mu update ########
            for(int t_idx=0; t_idx<time_horizon+1; t_idx++){
                x = x_list[t_idx];
                if(t_idx == time_horizon)
                    u = mat(U_DIM, 1);
                else
                    u = u_list[t_idx];
                const_vector = get_const(x, u, foot_pos_list[t_idx], contact_phi_list[t_idx]);
                lambda_list[t_idx] = (lambda_list[t_idx] + mu_vector*const_vector).max(0.0).min(max_lambda);
            }
            mu_vector = mu_vector*mu_scaling_factor;
            // ######################################

            // ######## backward pass ########
            x = x_list[time_horizon];
            u = mat(U_DIM, 1);
            const_vector = get_const(x, u, foot_pos_list[time_horizon], contact_phi_list[time_horizon]);
            C_x_mat = get_Cx(x, u, foot_pos_list[time_horizon], contact_phi_list[time_horizon]);
            C_u_mat = get_Cu(x, u, foot_pos_list[time_horizon], contact_phi_list[time_horizon]);
            I_mat = get_I(const_vector, lambda_list[time_horizon], mu_vector);
            P_mat = Qf_mat + C_x_mat.transpose().matmul(I_mat.matmul(C_x_mat));
            p_vector = Qf_mat.matmul(x - target_x_list[time_horizon]) + C_x_mat.transpose().matmul(lambda_list[time_horizon] + I_mat.matmul(const_vector));
            for(int t_idx=time_horizon-1; t_idx>=0; t_idx--){
                x = x_list[t_idx];
                u = u_list[t_idx];
                delta_time = delta_time_list[t_idx];
                A_mat = get_A(x, u, foot_pos_list[t_idx], delta_time);
                B_mat = get_B(x, u, foot_pos_list[t_idx], delta_time);
                const_vector = get_const(x, u, foot_pos_list[t_idx], contact_phi_list[t_idx]);
                C_x_mat = get_Cx(x, u, foot_pos_list[t_idx], contact_phi_list[t_idx]);
                C_u_mat = get_Cu(x, u, foot_pos_list[t_idx], contact_phi_list[t_idx]);
                I_mat = get_I(const_vector, lambda_list[t_idx], mu_vector);

                Qxx = Q_mat + A_mat.transpose().matmul(P_mat.matmul(A_mat)) + C_x_mat.transpose().matmul(I_mat.matmul(C_x_mat));
                Quu = R_mat + B_mat.transpose().matmul(P_mat.matmul(B_mat)) + C_u_mat.transpose().matmul(I_mat.matmul(C_u_mat));
                Qux = B_mat.transpose().matmul(P_mat.matmul(A_mat)) + C_u_mat.transpose().matmul(I_mat.matmul(C_x_mat));
                Qxu = A_mat.transpose().matmul(P_mat.matmul(B_mat)) + C_x_mat.transpose().matmul(I_mat.matmul(C_u_mat));
                Qx = Q_mat.matmul(x - target_x_list[t_idx]) + A_mat.transpose().matmul(p_vector) + C_x_mat.transpose().matmul(lambda_list[t_idx] + I_mat.matmul(const_vector));
                Qu = R_mat.matmul(u - target_u_list[t_idx]) + B_mat.transpose().matmul(p_vector) + C_u_mat.transpose().matmul(lambda_list[t_idx] + I_mat.matmul(const_vector));

                temp_mat = Quu + mat::eye(U_DIM)*damping_ratio;
                K_mat = temp_mat.inverse_matmul(Qux)*(-1.0);
                d_vector = temp_mat.inverse_matmul(Qu)*(-1.0);

                P_mat = Qxx + K_mat.transpose().matmul(Quu.matmul(K_mat) + Qux) + Qxu.matmul(K_mat);
                p_vector = Qx + K_mat.transpose().matmul(Quu.matmul(d_vector) + Qu) + Qxu.matmul(d_vector);
                K_list[t_idx] = K_mat;
                d_list[t_idx] = d_vector;
            }
            // ###############################

            // ######## forward pass ########
            temp_learning_rate = learning_rate;
            while(1){
                new_x_list[0] = init_x;
                J_value = 0.0;
                for(int t_idx=0; t_idx<time_horizon; t_idx++){
                    delta_x = new_x_list[t_idx] - x_list[t_idx];
                    delta_u = K_list[t_idx].matmul(delta_x) + d_list[t_idx]*learning_rate;
                    new_u_list[t_idx] = u_list[t_idx] + delta_u;
                    next_x = transition(new_x_list[t_idx], new_u_list[t_idx], foot_pos_list[t_idx], delta_time_list[t_idx]);
                    new_x_list[t_idx + 1] = next_x;

                    temp_vector = new_x_list[t_idx] - target_x_list[t_idx];
                    temp_mat = temp_vector.transpose().matmul(Q_mat.matmul(temp_vector))*0.5;
                    if(temp_mat.col != 1 || temp_mat.row != 1){
                        std::cerr<<"[error!!] do not match dimension"<<std::endl;
                        throw ERR;
                    }
                    J_value += temp_mat.data[0];
                    temp_vector = new_u_list[t_idx] - target_u_list[t_idx];
                    temp_mat = temp_vector.transpose().matmul(R_mat.matmul(temp_vector))*0.5;
                    if(temp_mat.col != 1 || temp_mat.row != 1){
                        std::cerr<<"[error!!] do not match dimension"<<std::endl;
                        throw ERR;
                    }
                    J_value += temp_mat.data[0];
                }
                temp_vector = new_x_list[time_horizon] - target_x_list[time_horizon];
                temp_mat = temp_vector.transpose().matmul(Qf_mat.matmul(temp_vector))*0.5;
                if(temp_mat.col != 1 || temp_mat.row != 1){
                    std::cerr<<"[error!!] do not match dimension"<<std::endl;
                    throw ERR;
                }
                J_value += temp_mat.data[0];
                if(pre_J_value - J_value >= 0.0)
                    break;
                learning_rate *= 0.5;
            }
            
            for(int i=0;i<time_horizon+1;i++){
                if(i < time_horizon){
                    u_list[i] = new_u_list[i];
                }
                x_list[i] = new_x_list[i];
            }
            // ##############################
        }

        temp_vector = K_list[0].matmul(new_x_list[0] - x_list[0]) + d_list[0] + u_list[0];
        memcpy(return_var, temp_vector.data, sizeof(double)*U_DIM);
        for(int i=0;i<time_horizon;i++){
            memcpy(&init_u_data[i*U_DIM], u_list[i].data, sizeof(double)*U_DIM);
        }

        delete[] x_list;
        delete[] u_list;
        delete[] new_x_list;
        delete[] new_u_list;
        delete[] target_x_list;
        delete[] target_u_list;
        delete[] lambda_list;
        delete[] K_list;
        delete[] d_list;
        delete[] foot_pos_list;
        delete[] contact_phi_list;

        return;
    }
}

mat transition(const mat & x, const mat & u, const mat & foot_pos_data, double delta_t){
    mat com_pos(&x.data[0], 3, 1);
    mat rpy(&x.data[3], 3, 1);
    mat com_vel(&x.data[6], 3, 1);
    mat base_angle_vel(&x.data[9], 3, 1);
    mat force_net(3, 1);
    mat base_torque_net(3, 1);
    mat approx_rot(3, 3);

    mat next_com_pos(3, 1);
    mat next_rpy(3, 1);
    mat next_com_vel(3, 1);
    mat next_base_ang_vel(3, 1);
    mat next_x(X_DIM, 1);

    mat force_list[NUM_LEG];
    mat foot_pos_list[NUM_LEG];

    approx_rot = mat::z_rot(rpy.data[2]);
    for(int i=0;i<NUM_LEG;i++){
        force_list[i] = mat(&u.data[i*3], 3, 1);
        foot_pos_list[i] = mat(&foot_pos_data.data[i*3], 3, 1);
        force_net = force_net + force_list[i];
        base_torque_net = base_torque_net + (foot_pos_list[i] - com_pos).cross(force_list[i]);
    }
    base_torque_net = approx_rot.transpose().matmul(base_torque_net);

    next_com_pos = com_pos + com_vel*delta_t + (force_net*(1/mass) + gravity)*(0.5*delta_t*delta_t);
    next_rpy = rpy + base_angle_vel*delta_t + inertia.inverse().matmul(base_torque_net)*(0.5*delta_t*delta_t);
    next_com_vel = com_vel + (force_net*(1/mass) + gravity)*delta_t;
    next_base_ang_vel = base_angle_vel + inertia.inverse().matmul(base_torque_net)*delta_t;

    memcpy(&next_x.data[0], next_com_pos.data, sizeof(double)*3);  
    memcpy(&next_x.data[3], next_rpy.data, sizeof(double)*3);  
    memcpy(&next_x.data[6], next_com_vel.data, sizeof(double)*3);  
    memcpy(&next_x.data[9], next_base_ang_vel.data, sizeof(double)*3);  
    return next_x;
}

mat get_const(const mat & x, const mat & u, const mat & foot_pos_data, const mat & contact_phi_list){
    mat com_pos(&x.data[0], 3, 1);
    mat rpy(&x.data[3], 3, 1);
    mat com_vel(&x.data[6], 3, 1);
    mat base_angle_vel(&x.data[9], 3, 1);

    double z_const = 0.3; //0.4 * 0.9;
    double friction_coeff = 0.9;
    double ground_normal_data[] = {0, 0, 1};
    mat ground_normal(ground_normal_data, 3, 1);

    mat heap_pos(3, 1);
    mat base_rot(3, 3);
    double leg_dist;

    mat const_vector(CONST_DIM, 1);
    int const_idx = 0;

    mat force_list[NUM_LEG];
    mat foot_pos_list[NUM_LEG];

    for(int i=0;i<NUM_LEG;i++){
        force_list[i] = mat(&u.data[i*3], 3, 1);
        foot_pos_list[i] = mat(&foot_pos_data.data[i*3], 3, 1);
    }

    // kinematic leg limits
    base_rot = mat::rpy_rot(rpy.data);
    for(int i=0;i<NUM_LEG;i++){
        heap_pos = base_rot.matmul(abduct_org_list[i]) + com_pos;
        leg_dist = (foot_pos_list[i] - heap_pos).magnitude();
        const_vector.data[const_idx] = (leg_dist - z_const)*contact_phi_list.data[i];
        const_idx++;
    }

    // swing leg's force limits
    const_vector.data[const_idx] = 0;
    for(int i=0;i<NUM_LEG;i++){
        const_vector.data[const_idx] += (1 - contact_phi_list.data[i])*force_list[i].magnitude();
    }
    const_idx ++;

    // positive ground force normal
    for(int i=0;i<NUM_LEG;i++){
        const_vector.data[const_idx] = -force_list[i].dot(ground_normal);
        const_idx++;
    }

    // friction pyramid
    for(int i=0;i<NUM_LEG;i++){
        const_vector.data[const_idx] = abs(force_list[i].data[0]) - friction_coeff*force_list[i].data[2];
        const_idx ++;
        const_vector.data[const_idx] = abs(force_list[i].data[1]) - friction_coeff*force_list[i].data[2];
        const_idx ++;
    }

    return const_vector;
}

mat get_A(const mat & x, const mat & u, const mat & foot_pos_data, double delta_t){
    mat next_x(X_DIM, 1);
    mat temp_x(X_DIM, 1);
    mat temp_next_x(X_DIM, 1);
    double eps = 0.001;
    double* data = new double[X_DIM*X_DIM];

    next_x = transition(x, u, foot_pos_data, delta_t);

    for(int i=0;i<X_DIM;i++){
        temp_x = x;
        temp_x.data[i] += eps;
        temp_next_x = transition(temp_x, u, foot_pos_data, delta_t);
        for(int j=0;j<X_DIM;j++){
            data[j*X_DIM + i] = (temp_next_x.data[j] - next_x.data[j])/eps;
        }
    }

    mat A(data, X_DIM, X_DIM);
    delete[] data;
    return A;
}

mat get_B(const mat & x, const mat & u, const mat & foot_pos_data, double delta_t){
    mat next_x(X_DIM, 1);
    mat temp_u(U_DIM, 1);
    mat temp_next_x(X_DIM, 1);
    double eps = 0.001;
    double* data = new double[X_DIM*U_DIM];

    next_x = transition(x, u, foot_pos_data, delta_t);

    for(int i=0;i<U_DIM;i++){
        temp_u = u;
        temp_u.data[i] += eps;
        temp_next_x = transition(x, temp_u, foot_pos_data, delta_t);
        for(int j=0;j<X_DIM;j++){
            data[j*U_DIM + i] = (temp_next_x.data[j] - next_x.data[j])/eps;
        }
    }

    mat B(data, X_DIM, U_DIM);
    delete[] data;
    return B;
}

mat get_Cx(const mat & x, const mat & u, const mat & foot_pos_data, const mat & contact_phi_list){
    mat next_const(CONST_DIM, 1);
    mat temp_x(X_DIM, 1);
    mat temp_next_const(CONST_DIM, 1);
    double eps = 0.001;
    double* data = new double[CONST_DIM*X_DIM];

    next_const = get_const(x, u, foot_pos_data, contact_phi_list);

    for(int i=0;i<X_DIM;i++){
        temp_x = x;
        temp_x.data[i] += eps;
        temp_next_const = get_const(temp_x, u, foot_pos_data, contact_phi_list);
        for(int j=0;j<CONST_DIM;j++){
            data[j*X_DIM + i] = (temp_next_const.data[j] - next_const.data[j])/eps;
        }
    }

    mat A(data, CONST_DIM, X_DIM);
    delete[] data;
    return A;
}

mat get_Cu(const mat & x, const mat & u, const mat & foot_pos_data, const mat & contact_phi_list){
    mat next_const(CONST_DIM, 1);
    mat temp_u(U_DIM, 1);
    mat temp_next_const(CONST_DIM, 1);
    double eps = 0.001;
    double* data = new double[CONST_DIM*U_DIM];

    next_const = get_const(x, u, foot_pos_data, contact_phi_list);

    for(int i=0;i<U_DIM;i++){
        temp_u = u;
        temp_u.data[i] += eps;
        temp_next_const = get_const(x, temp_u, foot_pos_data, contact_phi_list);
        for(int j=0;j<CONST_DIM;j++){
            data[j*U_DIM + i] = (temp_next_const.data[j] - next_const.data[j])/eps;
        }
    }

    mat A(data, CONST_DIM, U_DIM);
    delete[] data;
    return A;
}

mat get_I(const mat & const_vector, const mat & lambda_vector, const mat & mu_vector){
    double* data = new double[CONST_DIM*CONST_DIM];
    for(int i=0;i<CONST_DIM;i++){
        for(int j=0;j<CONST_DIM;j++){
            if(i != j){
                data[i*CONST_DIM + j] = 0;
                continue;
            }
            if(const_vector.data[i] < 0 || lambda_vector.data[i] == 0.0)
                data[i*CONST_DIM + j] = 0;
            else
                data[i*CONST_DIM + j] = mu_vector.data[i];
        }
    }
    mat A(data, CONST_DIM, CONST_DIM);
    delete[] data;
    return A;
}