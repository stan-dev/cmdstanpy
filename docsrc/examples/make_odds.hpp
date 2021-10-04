#include <boost/math/tools/promotion.hpp>
#include <ostream>

namespace bernoulli_external_model_namespace
{
    template <typename T0__>
    inline typename boost::math::tools::promote_args<T0__>::type make_odds(const T0__ &
                                                                               theta,
                                                                           std::ostream *pstream__)
    {
        return theta / (1 - theta);
    }
}