"use strict ";


var app = new Vue({
    el: '#xiaoshi',
    data: {
        modern: "",
        messages: [],
    },
    methods: {
        sendMessage: function() {
            if (this.modern.length < 10) {
                alert("至少给人家写十个字啦。")
                return;
            }
            vm = this;
            axios.post('/send', {
                modern: this.modern,
            }).then(function(response) {
                vm.messages.push(response.data);
            });
            this.modern = "";
        },
    },
});