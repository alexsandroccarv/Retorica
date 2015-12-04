# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from fake_useragent import UserAgent


def random_user_agent():
    """Returns a *convincingly valid* random User-Agent.
    """
    return UserAgent().random


class RandomUserAgentMiddleware(object):
    """This middleware sets the User-Agent to some random value for each
    request. The generated values also look kind of convincingly valid.
    """

    def process_request(self, request, spider):
        request.headers['User-Agent'] = random_user_agent()
