import launch

if not launch.is_installed("yaml"):
    launch.run_pip("install pyyaml", "pyyaml for Infinity Grid Script")
if not launch.is_installed("colorama"):
    launch.run_pip('install colorama', "making sure logging is more descriptive")
