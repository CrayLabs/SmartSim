#!groovy

node('cicero') {
  def STAGE = ''

  try {
    notifyBuild('STARTED')

    //env.NAME = ""
    //env.IYUM_REPO_NAME = "${env.NAME}"
    env.IYUM_REPO_NAME = "analytics"
    env.IYUM_REPO_PREFIX = "poseidon"

    //
    // Stage Prep
    //
    STAGE = 'Prep'
    stage "${STAGE}"

    sh "rm -rf build_rpm_dir"

    //
    // Checkout
    //
    STAGE = 'Checkout'
    stage "${STAGE}"

    // Checkout code from repository and update any submodules
    checkout scm

    //
    // Build
    // 
    STAGE = 'Build'
    stage "${STAGE}"

    // save environment variables
    sh 'env | sort ; env | sort > env.txt'

    echo """
    BRANCH_NAME=${env.BRANCH_NAME}
    BUILD_ID=${env.BUILD_ID}
    BUILD_NUMBER=${env.BUILD_NUMBER}
    BUILD_TAG=${env.BUILD_TAG}
    BUILD_URL=${env.BUILD_URL}
    HOME=${env.HOME}
    JENKINS_URL=${env.JENKINS_URL}
    JOB_BASE_NAME=${env.JOB_BASE_NAME}
    JOB_NAME=${env.JOB_NAME}
    JOB_URL=${env.JOB_URL}
    WORKSPACE=${env.WORKSPACE}
    """

    // build docs and deploy
    sh "./build.sh"

    //
    // Package
    //
    STAGE = 'Package'
    stage "${STAGE}"
    // TODO SET FOR WORKFLOWS
    sh "mkdir build_rpm_dir && cd build_rpm_dir && ../build_rpm"

    //
    // Archive
    // 
    STAGE = 'Archive'
    stage "${STAGE}"

    //tell Jenkins to archive the apks
    step([$class: 'ArtifactArchiver', artifacts: 'build_rpm_dir/*.rpm', fingerprint: true])
    //
    // Upload
    //
    STAGE = 'Upload'
    stage "${STAGE}"

    sh "source build_rpm_dir/build_common.shrc && transfer_rpms build_rpm_dir/*.rpm"


    //
    // Notify
    //
    STAGE = 'Notify'
    stage "${STAGE}"

    sh "source build_rpm_dir/build_common.shrc && notify build_rpm_dir/*.rpm"

  } catch (e) {
    // If an exception is thrown then the build fails
    currentBuild.result = "FAILED"
    sh "source build/build_common.shrc && notify fail Failed at stage: ${STAGE}"
    throw e

  } finally {
    // Send out a notification whether the build is successful or it fails
    notifyBuild(currentBuild.result)
  }
}

def notifyBuild(String buildStatus = 'STARTED') {
  // build status of null means successful
  buildStatus =  buildStatus ?: 'SUCCESSFUL'

  // Default values
  def colorName = 'YELLOW'
  def colorCode = '#FFFF00'
  def subject = "${buildStatus}: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'"
  def summary = "${subject} <a href=${env.BUILD_URL}>(View Build)</a>"

  // Override default values based on build status
  if (buildStatus == 'STARTED') {
    color = 'PURPLE'
    colorCode = '#800080'
  } else if (buildStatus == 'SUCCESSFUL') {
    color = 'GREEN'
    colorCode = '#00FF00'
  } else {
    color = 'RED'
    colorCode = '#FF0000'
  }
}