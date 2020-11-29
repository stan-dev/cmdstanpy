import sys

import requests

# This scrip accepts 4 arguments in order:
# Project name in readthedocs, readthedocs username, readthedocs password and version in readthedocs to set as default.
# Example:
# python3 change_default_version.py cmdstanpy snick secretpassword stable-0.9.66


def get_csrf_token(cookies):
    if 'csrftoken' in cookies:
        # Django 1.6 and up
        return cookies['csrftoken']
    else:
        # older versions
        return cookies['csrf']


def set_version_active_on_rtd(
    project_slug,
    rtd_user,
    rtd_password,
    default_version,
    server_addr="https://readthedocs.org",
):

    with requests.session() as s:

        url = server_addr + "/accounts/login/"

        # Fetch the login page
        s.get(url)
        # Extract CSRF token
        csrftoken = get_csrf_token(s.cookies)

        # Build out login data
        login_data = dict(
            login=rtd_user,
            password=rtd_password,
            csrfmiddlewaretoken=csrftoken,
            next='/',
        )
        # Post login request
        r = s.post(url, data=login_data, headers=dict(Referer=url))

        # Set url for dashboard/advanced
        url = server_addr + "/dashboard/" + project_slug + "/advanced/"
        # Extract CSRF token
        csrftoken = get_csrf_token(s.cookies)

        # Build out version data
        version_data = {
            'csrfmiddlewaretoken': csrftoken,
            'default_version': default_version,
            'default_branch': 'develop',
            'analytics_code': '',
            'documentation_type': 'sphinx',
            'requirements_file': '',
            'python_interpreter': 'python',
            'conf_py_file': 'docsrc/conf.py',
            'enable_pdf_build': 'on',
            'enable_epub_build': 'on',
            'save': 'Save',
        }

        # Post our version data
        r = s.post(url, data=version_data, headers=dict(Referer=url))


if __name__ == '__main__':
    set_version_active_on_rtd(
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    )
