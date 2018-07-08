/*
 *       /\        Matteo Cicuttin (C) 2016, 2017, 2018
 *      /__\       matteo.cicuttin@enpc.fr
 *     /_\/_\      École Nationale des Ponts et Chaussées - CERMICS
 *    /\    /\
 *   /__\  /__\    DISK++, a template library for DIscontinuous SKeletal
 *  /_\/_\/_\/_\   methods.
 *
 * This file is copyright of the following authors:
 * Nicolas Pignet  (C) 2018                     nicolas.pignet@enpc.fr
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * If you use this code or parts of it for scientific publications, you
 * are required to cite it as following:
 *
 * Implementation of Discontinuous Skeletal methods on arbitrary-dimensional,
 * polytopal meshes using generic programming.
 * M. Cicuttin, D. A. Di Pietro, A. Ern.
 * Journal of Computational and Applied Mathematics.
 * DOI: 10.1016/j.cam.2017.09.017
 */

#pragma once

#include "common/eigen.hpp"
#include "mechanics/behaviors/maths_tensor.hpp"
#include "mechanics/behaviors/maths_utils.hpp"
#include "mesh/point.hpp"

#define _USE_MATH_DEFINES
#include <cmath>

namespace disk
{

// Law for LinearLaw (test of finite deformations)

template<typename scalar_type>
class LinearLaw_Data
{
  private:
    scalar_type m_lambda;
    scalar_type m_mu;

  public:
    LinearLaw_Data() : m_lambda(1.0), m_mu(1.0) {}

    LinearLaw_Data(const scalar_type& lambda, const scalar_type& mu) : m_lambda(lambda), m_mu(mu) {}

    scalar_type
    getE() const
    {
        return m_mu * (3 * m_lambda + 2 * m_mu) / (m_lambda + m_mu);
    }

    scalar_type
    getNu() const
    {
        return m_lambda / (2 * (m_lambda + m_mu));
    }

    scalar_type
    getLambda() const
    {
        return m_lambda;
    }

    scalar_type
    getMu() const
    {
        return m_mu;
    }

    void
    print() const
    {
        std::cout << "Material parameters: " << std::endl;
        std::cout << "* E: " << getE() << std::endl;
        std::cout << "* Nu: " << getNu() << std::endl;
        std::cout << "* Lambda: " << getLambda() << std::endl;
        std::cout << "* Mu: " << getMu() << std::endl;
    }
};

// Input : symetric stain tensor(Gs)

//           dev = normL2(Gs - trace(Gs) / dim * Id)

//             Stress : sigma = 2 *\tilde{mu}(dev(Gs)) * Gs + \tilde{lambda}(dev(Gs)) * trace(Gs) * Id
// \tilde{mu}(dev(Gs)) = mu * (1 + (1 + dev(Gs)) ^ {-1 / 2})
// \tilde{lambda}(dev(Gs)) = ((lambda + mu / 2) - mu / 2 * (1 + dev(Gs)) ^ {-1 / 2})

//                          Tangent Moduli : C = 2 * mu * I4 + lambda * prod_Kronecker(Id, Id) /
//                                                               it is the elastic moduli

template<typename scalar_type, int DIM>
class LinearLaw_qp
{
    typedef static_matrix<scalar_type, DIM, DIM> static_matrix_type;
    typedef static_matrix<scalar_type, 3, 3>     static_matrix_type3D;
    typedef LinearLaw_Data<scalar_type>               data_type;

    static_matrix_type zero_matrix = static_matrix_type::Zero();
    static_matrix_type3D zero_matrix3D = static_matrix_type3D::Zero();

    // coordinat and weight of considered gauss point.
    point<scalar_type, DIM> m_point;
    scalar_type             m_weight;

    // internal variables at previous step
    static_matrix_type m_estrain_prev; // elastic strain

    // internal variables at current step
    static_matrix_type m_estrain_curr; // elastic strain

    static_tensor<scalar_type, DIM>
    elastic_modulus(const data_type& data) const
    {

        return 2 * data.getMu() * compute_IdentitySymTensor<scalar_type, DIM>() +
               data.getLambda() * compute_IxI<scalar_type, DIM>();
    }

    static_matrix_type3D
    convert3D(const static_matrix_type& mat) const
    {
        static_matrix_type3D ret  = zero_matrix;
        ret.block(0, 0, DIM, DIM) = mat;

        return ret;
    }

  public:
    LinearLaw_qp(const point<scalar_type, DIM>& point, const scalar_type& weight) :
      m_point(point), m_weight(weight), m_estrain_prev(zero_matrix), m_estrain_curr(zero_matrix)
    {
    }

    point<scalar_type, DIM>
    point() const
    {
        return m_point;
    }

    scalar_type
    weight() const
    {
        return m_weight;
    }

    bool
    is_plastic() const
    {
        return false;
    }

    static_matrix_type3D
    getElasticStrain() const
    {
        return convert3D(m_estrain_curr);
    }

    static_matrix_type3D
    getPlasticStrain() const
    {
        return zero_matrix3D;
    }

    static_matrix_type
    getTotalStrain() const
    {
        return m_estrain_curr;
    }

    static_matrix_type
    getTotalStrainPrev() const
    {
        return m_estrain_prev;
    }

    scalar_type
    getAccumulatedPlasticStrain() const
    {
        return scalar_type(0);
    }

    void
    update()
    {
        m_estrain_prev = m_estrain_curr;
    }

    static_matrix_type
    compute_stress(const data_type& data) const
    {
        return data.getLambda() * m_estrain_curr;
    }

    std::pair<static_matrix_type, static_tensor<scalar_type, DIM>>
    compute_whole(const static_matrix_type& incr_F, const data_type& data, bool tangentmodulus = true)
    {
        static_tensor<scalar_type, DIM> Cep = data.getLambda() * compute_IdentityTensor<scalar_type, DIM>();

        // is always elastic
        m_estrain_curr = m_estrain_prev + incr_F;

        // compute Cauchy stress
        const static_matrix_type stress = this->compute_stress(data);

        return std::make_pair(stress, Cep);
    }
};
}