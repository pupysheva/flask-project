
function load() {
  let eventSource = new EventSource("/get_recommendation/" + window.location.pathname.split('/')[1] + "/load");

  eventSource.onmessage = function(event) {
    console.log("Новое сообщение", event.data);
  }; 
}
