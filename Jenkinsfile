#!/usr/bin/env groovy

def checkoutBranch(branch){
    checkout([$class: 'GitSCM',
              branches: [[name: '*/' + branch]],
              doGenerateSubmoduleConfigurations: false,
              extensions: [],
              submoduleCfg: [],
              userRemoteConfigs: [[url: "https://github.com/stan-dev/cmdstanpy.git", credentialsId: 'a630aebc-6861-4e69-b497-fd7f496ec46b']]]
    )
}

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
        stage("Create release branch") {
            steps{
                deleteDir()
                checkoutBranch("develop")

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

                        # Git identity
                        git config --global auth.token ${GITHUB_TOKEN}

                        # Push new release branch
                        git add .
                        git commit -m "release/v${params.new_version}: updating version numbers" -a
                        git push https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/stan-dev/cmdstanpy.git release/v${params.new_version}
                    """
                }
            }
        }

        stage("Merge into develop") {
            steps{
                deleteDir()
                checkoutBranch("develop")

                withCredentials([usernamePassword(credentialsId: 'a630aebc-6861-4e69-b497-fd7f496ec46b', usernameVariable: 'GIT_USERNAME', passwordVariable: 'GIT_PASSWORD')]) {
                    sh """#!/bin/bash
                        git pull origin develop
                        git merge release/v${params.new_version}

                        git config --global auth.token ${GITHUB_TOKEN}
                        git push https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/stan-dev/cmdstanpy.git develop
                        
                        git branch -d release/v${params.new_version}
                    """
                }
            }
        }

        stage("Tag version") {
            steps{
                deleteDir()
                checkoutBranch("develop")

                withCredentials([usernamePassword(credentialsId: 'a630aebc-6861-4e69-b497-fd7f496ec46b', usernameVariable: 'GIT_USERNAME', passwordVariable: 'GIT_PASSWORD')]) {
                    sh """#!/bin/bash
                        git checkout develop
                        git pull origin develop --ff
                        git tag -a "v${params.new_version}" -m "Tagging v${params.new_version}"

                        git config --global auth.token ${GITHUB_TOKEN}

                        git push https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/stan-dev/cmdstanpy.git tag v${params.new_version}
                    """
                }
            }
        }

        stage("Update master branch to new version") {
            steps{
                deleteDir()
                checkoutBranch("master")
                
                withCredentials([usernamePassword(credentialsId: 'a630aebc-6861-4e69-b497-fd7f496ec46b', usernameVariable: 'GIT_USERNAME', passwordVariable: 'GIT_PASSWORD')]) {
                    /* Update master branch to the new version */
                    sh """#!/bin/bash
                        git reset --hard v${params.new_version}
                        git config --global auth.token ${GITHUB_TOKEN}
                        git push https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/stan-dev/cmdstanpy.git master
                    """
                }
            }
        }

        stage("Upload package to pypi") {
            steps{
                deleteDir()
                checkoutBranch("master")

                withCredentials([usernamePassword(credentialsId: 'pypi-snick-token', usernameVariable: 'PYPI_USERNAME', passwordVariable: 'PYPI_TOKEN')]) {
                    sh """#!/bin/bash

                        # Install python dependencies
                        pip3 install --no-cache-dir --upgrade pip
                        pip3 install --no-cache-dir twine wheel
                        pip3 install --no-cache-dir -r requirements.txt

                        # Build wheel
                        python3 setup.py bdist_wheel
                        pip3 install --no-cache-dir dist/*.whl

                        # Upload wheels
                        python3 -m twine upload -u __token__ -p ${PYPI_TOKEN} --skip-existing dist/*
                    """
                }
            }
        }

        stage("Change ReadTheDocs default version") {
            steps{
                deleteDir()
                checkoutBranch("master")
                withCredentials([usernamePassword(credentialsId: 'readthedocs-snick-username-password', usernameVariable: 'RTD_USERNAME', passwordVariable: 'RTD_PASSWORD')]) {
                    sh """#!/bin/bash
                        python change_default_version.py cmdstanpy ${RTD_USERNAME} ${RTD_PASSWORD} v${params.new_version}
                    """
                }
            }
        }

    }
}