// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App.vue'
import router from './router'

// Import Cesium and set token before anything else
import { Ion } from 'cesium'

// Importing Bootstrap Vue
import BootstrapVue from 'bootstrap-vue'
import 'bootstrap/dist/css/bootstrap.min.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'

// Using imported components
import VueRouter from 'vue-router'

// Set the Cesium token directly
// This is a hardcoded token for development only
// In production, you would use environment variables
const cesiumToken = [
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9',
    'eyJqdGkiOiI1ZjdjYmVkNi1mNTcwLTQ4MjYtYjU5Yi03OGZhMDZkMzk0NjgiLCJpZCI6MzAzMTY5LCJpYXQiOjE3NDgzNjc3Mzh9',
    'qzCis71TVJZRDHsO3eLzKbGAUW0zTjKS-M6L3zGzxII'
].join('.')
Ion.defaultAccessToken = cesiumToken
console.log('Cesium token set directly')

Vue.use(VueRouter)
Vue.use(BootstrapVue)

Vue.config.productionTip = false

Vue.prototype.$eventHub = new Vue() // Global event bus

/* eslint-disable no-new */
new Vue({
    el: '#app',
    router,
    components: { App },
    template: '<App/>'
})
