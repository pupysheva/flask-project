function check_progress(url, method = $.get) {
    let progress_bar_dom = document.getElementById("myBar");
    method(url, function(task) {
        percent_cache = 0.0
        check_progress_args(task, (percent) =>
        {
            if (percent != percent_cache)
            {
                progress_bar_dom.style.width = (percent * 100) + "%"
                percent_cache = percent
            }
        })
    })
    function check_progress_args(task_id, progress_bar_callback) {
        function worker() {
            $.get('progress/' + task_id, function(progress) {
                progress_bar_callback(progress)
                if (progress < 1) {
                    setTimeout(worker, 199)
                }
            })
        }
        worker()
    }
}

function load() {
  let eventSource = new EventSource("/get_recommendation/" + window.location.pathname.split('/')[1] + "/load");

  eventSource.onmessage = function(event) {
    console.log("Новое сообщение", event.data);
  }; 
}
