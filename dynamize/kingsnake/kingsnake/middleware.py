# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from scrapy import signals
from fake_useragent import UserAgent


def random_user_agent():
    """Returns a *convincingly valid* random User-Agent.
    """
    return UserAgent().random


def rsetdefaults(struct, defaults):
    for (k, v) in defaults.items():
        struct.setdefault(k, v)


class RandomUserAgentMiddleware(object):
    """This middleware sets the User-Agent to some random value for each
    request. The generated values also look kind of convincingly valid.
    """

    def process_request(self, request, spider):
        ua = request.headers.pop('User-Agent', ['Scrapy'])[0]

        if ua.lower().startswith('scrapy'):
            ua = random_user_agent()

        request.headers.update({
            'User-Agent': ua,
            #'Pragma': 'no-cache',
            'Host': 'www.camara.gov.br',
            #'Cache-Control': 'no-cache',
            'Accept-Language': 'pt-BR,pt;q=0.8,en-US;q=0.6,en;q=0.4',
        })

        request.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, sdch',
            'Accept-Language': 'pt-BR,pt;q=0.8,en-US;q=0.6,en;q=0.4',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Pragma': 'no-cache',
            'Cookie': '__utma=260181407.86875399.1416311980.1416311980.1416333000.2; __utmz=260181407.1416333000.2.2.utmcsr=labhackercd.net|utmccn=(referral)|utmcmd=referral|utmcct=/; BALANCEID=mycluster.node1'
        })
