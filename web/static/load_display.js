function check_progress(url) {
    let progress_bar_dom = document.getElementById("myBar");
    $.get(url, function(task) {
        check_progress_args(task, (percent) => { progress_bar_dom.style.width = percent + "%" })
    })
    function check_progress_args(task_id, progress_bar_callback) {
        function worker() {
            $.get('progress/' + task_id, function(progress) {
                progress_bar_callback(progress)
                if (progress < 100) {
                    setTimeout(worker, 1000)
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
