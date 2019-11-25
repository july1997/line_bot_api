new Vue({
    el: '#app',
    delimiters: ['[[', ']]'],
    data: {
        input: '',
        checked: false,
        list: [

        ]
    },
    methods: {
        addItem: function() {
            this.list.push({ user: 'あなた', message: this.input.replace(/[\r?\n]*\s+/g, '') })
            this.getApi()
            this.input = ''
            this.checkList()
        },
        submit: function() {
            if(this.checked && this.input.replace(/[\r?\n]*\s+/g, '') != ''){
                this.list.push({ user: 'あなた', message: this.input.replace(/[\r?\n]*\s+/g, '') })
                this.getApi()
                this.input = ''
                this.checkList()
            }
        },
        getApi: function() {
            let params = new URLSearchParams();
            params.append('text', this.input);

            axios.post('../api/chat', params)
            .then(response => {
                this.list.push({ user: 'Bot', message: response.data })
                this.checkList()
            }).catch(error => {
                console.log(error);
            });
        },
        checkList: function(){
            if(this.list.length > 5){
                this.list.splice(0, 1);
            }
        }
    },
    computed: {
        getListFirstOne() {
            return this.list.slice(0,1)
        },
        getList() {
            return this.list.slice(1,this.list.length)
        }
    }
  })