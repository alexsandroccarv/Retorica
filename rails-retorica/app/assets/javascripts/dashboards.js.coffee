#= require d3
#= require lodash
#= require d3.forceLayout
#= require sprintf

# Quick hack to export a function to the *global* namespace so
# that it can be called from within script blocks
root = exports ? this

doTheThing = root.doTheThing = (source) ->

  $.getJSON(source).done (data) ->

    d3.select('.loading')
      .transition()
      .duration(1000)
      .style('opacity','0')
      .delay(500)
      .remove()

    topics = _(data.topics)

    topics = topics.filter (d) ->
      (not d.ignore) and d.title.indexOf("Solenidades") != 0

    # XXX FIXME ROFL LOL WTF Some topics have no emphases and nobody doesn't
    # know why or how. This is be causin' all sorts of trouble, like a bunch
    # of errors about images, <g>s and <node>s. So, please, please, don't
    # remove this if you are not really, really sure about what you're doing.
    # Thanks.
    topics = topics.filter (d) ->
      d.emphases?.length > 0

    item_id_counter = 0

    topics = topics.map (d, i) ->

      emphases = _(d.emphases)
      emphases = emphases.map (d, i) ->

        r =
          # XXX Ids MUST be unique through the entire visualization, or bad
          # things will happen.
          id: (item_id_counter++)
          value: d.emphasis
          author: d.name

        if d.deputado

          r.author = d.deputado.nome_parlamentar
          r.situacao = d.deputado.situacao
          r.sexo = d.deputado.sexo
          r.url = d.deputado.site_deputado
          r.email = d.deputado.email
          r.foto = d.deputado.foto_url
          r.uf = d.deputado.unidade_federativa[0]?.sigla
          r.partido = d.deputado.partidos[0]?.sigla


        return r

      emphases = emphases.value()

      return {
        id: d._id["$oid"],
        topic: d.title
        value: d3.sum(emphases, (d, i) -> d.value)
        children: emphases
      }

    topics = topics.value()

    details = null

    if not data.description
      details = data.title
    else
      sdata = {topic_count: topics.length}
      details = sprintf(data.description, sdata)

    d3.select('.intro .detail').text(details)

    d3.custom.forceLayout(topics)
