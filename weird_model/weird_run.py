import cmdstanpy
import platform

if platform.system() == "Windows":
    cmdstanpy.utils.cxx_toolchain_path()

program_code = """
parameters{
    real lambda;
    real theta;
}

model {
    real log_q;//lp of the first model;
    theta~normal(0,1); // model 1
    log_q=target();
    target+=-target();
    real log_p;//lp of the alternative model
    theta~normal(5,3); // model 2
    log_p=target();
    target+=target()*(lambda)-target() + (1-lambda)*log_q;
    lambda~beta(5,5);
    print("local_vars:", log_q, ",", log_p, ",", target());
}
"""

stan_file = "./stan_weird_example.stan"

with open(stan_file, "w") as f:
    print(program_code, file=f)

model = cmdstanpy.CmdStanModel(stan_file=stan_file)
fit = model.sample(chains=1, iter_warmup=100, iter_sampling=100)


print(fit.diagnose())


print("WEIRD MODEL 1 CHAIN --> COUNTING PRINTS")

# data
for i, fpath in enumerate(fit.runset.stdout_files):
    print("FILE {}: ".format(i), end=" ")
    with open(fpath, "r") as f:
        line_count = 0
        #lines = []
        for j, line in enumerate(f):
            if line.startswith("local_vars:"):
                line_count += 1
                #lines.append([float(item.strip()) for item in line.strip("local_vars:").split(",")])
        # data.append(lines)
        print("\n    print count:", line_count, "\n    total linecount", j, "\n\n")


fit2 = model.sample(chains=1, iter_warmup=100, iter_sampling=100)


print(fit2.diagnose())

print("WEIRD MODEL 10 CHAIN --> COUNTING PRINTS")

# data
for i, fpath in enumerate(fit2.runset.stdout_files, 1):
    print("FILE {}: ".format(i), end=" ")
    with open(fpath, "r") as f:
        line_count = 0
        #lines = []
        for j, line in enumerate(f, 1):
            if line.startswith("local_vars:"):
                line_count += 1
                #lines.append([float(item.strip()) for item in line.strip("local_vars:").split(",")])
        # data.append(lines)
        print("\n    print count:", line_count, "\n    total linecount", j, "\n\n")
