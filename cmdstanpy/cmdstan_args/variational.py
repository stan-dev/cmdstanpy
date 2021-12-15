"""Arguments for the variational subcommand"""

from typing import List, Optional


class VariationalArgs:
    """Arguments needed for variational method."""

    VARIATIONAL_ALGOS = {'meanfield', 'fullrank'}

    def __init__(
        self,
        algorithm: Optional[str] = None,
        iter: Optional[int] = None,
        grad_samples: Optional[int] = None,
        elbo_samples: Optional[int] = None,
        eta: Optional[float] = None,
        adapt_iter: Optional[int] = None,
        adapt_engaged: bool = True,
        tol_rel_obj: Optional[float] = None,
        eval_elbo: Optional[int] = None,
        output_samples: Optional[int] = None,
    ) -> None:
        self.algorithm = algorithm
        self.iter = iter
        self.grad_samples = grad_samples
        self.elbo_samples = elbo_samples
        self.eta = eta
        self.adapt_iter = adapt_iter
        self.adapt_engaged = adapt_engaged
        self.tol_rel_obj = tol_rel_obj
        self.eval_elbo = eval_elbo
        self.output_samples = output_samples

    def validate(
        self, chains: Optional[int] = None  # pylint: disable=unused-argument
    ) -> None:
        """
        Check arguments correctness and consistency.
        """
        if (
            self.algorithm is not None
            and self.algorithm not in self.VARIATIONAL_ALGOS
        ):
            raise ValueError(
                'Please specify variational algorithms as one of [{}]'.format(
                    ', '.join(self.VARIATIONAL_ALGOS)
                )
            )
        if self.iter is not None:
            if self.iter < 1 or not isinstance(self.iter, int):
                raise ValueError(
                    'iter must be a positive integer,'
                    ' found {}'.format(self.iter)
                )
        if self.grad_samples is not None:
            if self.grad_samples < 1 or not isinstance(self.grad_samples, int):
                raise ValueError(
                    'grad_samples must be a positive integer,'
                    ' found {}'.format(self.grad_samples)
                )
        if self.elbo_samples is not None:
            if self.elbo_samples < 1 or not isinstance(self.elbo_samples, int):
                raise ValueError(
                    'elbo_samples must be a positive integer,'
                    ' found {}'.format(self.elbo_samples)
                )
        if self.eta is not None:
            if self.eta < 0 or not isinstance(self.eta, (int, float)):
                raise ValueError(
                    'eta must be a non-negative number,'
                    ' found {}'.format(self.eta)
                )
        if self.adapt_iter is not None:
            if self.adapt_iter < 1 or not isinstance(self.adapt_iter, int):
                raise ValueError(
                    'adapt_iter must be a positive integer,'
                    ' found {}'.format(self.adapt_iter)
                )
        if self.tol_rel_obj is not None:
            if self.tol_rel_obj <= 0 or not isinstance(
                self.tol_rel_obj, (int, float)
            ):
                raise ValueError(
                    'tol_rel_obj must be a positive number,'
                    ' found {}'.format(self.tol_rel_obj)
                )
        if self.eval_elbo is not None:
            if self.eval_elbo < 1 or not isinstance(self.eval_elbo, int):
                raise ValueError(
                    'eval_elbo must be a positive integer,'
                    ' found {}'.format(self.eval_elbo)
                )
        if self.output_samples is not None:
            if self.output_samples < 1 or not isinstance(
                self.output_samples, int
            ):
                raise ValueError(
                    'output_samples must be a positive integer,'
                    ' found {}'.format(self.output_samples)
                )

    # pylint: disable=unused-argument
    def compose(self, idx: int, cmd: List[str]) -> List[str]:
        """
        Compose CmdStan command for method-specific non-default arguments.
        """
        cmd.append('method=variational')
        if self.algorithm is not None:
            cmd.append('algorithm={}'.format(self.algorithm))
        if self.iter is not None:
            cmd.append('iter={}'.format(self.iter))
        if self.grad_samples is not None:
            cmd.append('grad_samples={}'.format(self.grad_samples))
        if self.elbo_samples is not None:
            cmd.append('elbo_samples={}'.format(self.elbo_samples))
        if self.eta is not None:
            cmd.append('eta={}'.format(self.eta))
        cmd.append('adapt')
        if self.adapt_engaged:
            cmd.append('engaged=1')
            if self.adapt_iter is not None:
                cmd.append('iter={}'.format(self.adapt_iter))
        else:
            cmd.append('engaged=0')
        if self.tol_rel_obj is not None:
            cmd.append('tol_rel_obj={}'.format(self.tol_rel_obj))
        if self.eval_elbo is not None:
            cmd.append('eval_elbo={}'.format(self.eval_elbo))
        if self.output_samples is not None:
            cmd.append('output_samples={}'.format(self.output_samples))
        return cmd
