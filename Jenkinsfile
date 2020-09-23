#!/usr/bin/env groovy

pipeline {
    agent { label 'linux-ec2' }
    options {
        skipDefaultCheckout()
        preserveStashes(buildCount: 5)
    }
    parameters {
        string(defaultValue: '0.9.67', name: 'new_version', description: "Version to release.")
        string(defaultValue: '0.9.66', name: 'old_version', description: "Old version to be replaced in cmdstanpy/_version")
    }
    environment {
        GITHUB_TOKEN = credentials('6e7c1e8f-ca2c-4b11-a70e-d934d3f6b681')
    }
    stages {
        stage("Activate version") {
            steps{
                deleteDir()

                checkout([$class: 'GitSCM',
                          branches: [[name: '*/develop']],
                          doGenerateSubmoduleConfigurations: false,
                          extensions: [],
                          submoduleCfg: [],
                          userRemoteConfigs: [[url: "https://github.com/stan-dev/cmdstanpy.git", credentialsId: 'a630aebc-6861-4e69-b497-fd7f496ec46b']]]
                )

                /* Create a new branch */
                withCredentials([usernamePassword(credentialsId: 'a630aebc-6861-4e69-b497-fd7f496ec46b', usernameVariable: 'GIT_USERNAME', passwordVariable: 'GIT_PASSWORD')]) {

                    /* Create release branch, change cmdstanpy/_version and generate docs. */
                    sh """#!/bin/bash

                        # Create new release branch
                        git checkout -b release/v${params.new_version}

                        # Change version in _version
                        sed -i 's/${params.old_version}/${params.new_version}/g' cmdstanpy/_version.py

                        # Generate docs
                        cd docsrc
                        make github
                        cd ..

                        git config --global user.email "mc.stanislaw@gmail.com"
                        git config --global user.name "Stan Jenkins"
                        git config --global auth.token ${GITHUB_TOKEN}

                        # Push new release branch
                        git add .
                        git commit -m "release/v${params.new_version}: updating version numbers" -a
                        git push https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/stan-dev/cmdstanpy.git release/v${params.new_version}
                    """

                    /* Merge into develop */
                    //sh """#!/bin/bash
                    //    git checkout develop
                    //    git pull origin develop
                    //    git merge release/v${params.new_version}
                    //    git push origin develop
                    //    git branch -d release/v${params.new_version}
                    //"""

                    /* Tag version */
                    sh """#!/bin/bash
                        git checkout develop
                        git pull origin develop --ff
                        git tag -a "v${params.new_version}" -m "Tagging v${params.new_version}"
                        git push origin "v${params.new_version}"
                    """

                    /* Update master branch to the new version */
                    //sh """#!/bin/bash
                    //    git checkout master
                    //    git reset --hard v${params.new_version}
                    //    git push origin master
                    //"""
                }

                withCredentials([usernamePassword(credentialsId: 'pypi-snick-token', usernameVariable: 'PYPI_USERNAME', passwordVariable: 'PYPI_TOKEN')]) {
                    /* Upload to PyPi */
                    sh """#!/bin/bash

                        # Install python dependencies
                        pip install --no-cache-dir --upgrade pip
                        pip install --no-cache-dir twine wheel
                        pip install --no-cache-dir -r requirements.txt

                        # Build wheel
                        python setup.py bdist_wheel
                        pip install --no-cache-dir dist/*.whl

                        # Upload wheels
                        twine upload -u __token__ -p ${PYPI_TOKEN} --skip-existing dist/*

                    """
                }

                withCredentials([usernamePassword(credentialsId: 'readthedocs-snick-username-password', usernameVariable: 'RTD_USERNAME', passwordVariable: 'RTD_PASSWORD')]) {
                    /* Set default version in readthedocs */
                    sh """#!/bin/bash
                        python change_default_version.py cmdstanpy ${RTD_USERNAME} ${RTD_PASSWORD} ${params.new_version}
                    """
                }

            }
        }
    }
}