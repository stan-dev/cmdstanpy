#include <ostream>

double make_odds(const double& theta, std::ostream *pstream__) {
  return theta / (1 - theta);
}
