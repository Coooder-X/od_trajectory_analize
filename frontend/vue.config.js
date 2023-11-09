const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  lintOnSave: false,
  devServer: {
    open: true,
    host: 'localhost',
    port: 8080,
    https: false,
    //以上的ip和端口是我们本机的;下面为需要跨域的
    proxy: {//配置跨域
        // 这个地方的 '/api'  名字要和底下 '^/api'  这个地方的名字一样。如果这里是 '/bpi'，那么底下就也要是 '^/bpi'
        '/api': {
            // target: 'http://localhost:5000',//这里后台的地址模拟的;应该填写你们真实的后台接口
            target: 'http://10.105.16.22:5000',
            // target: 'http://localhost:5050',
            // target: 'http://localhost:18082',
            ws: true,
            changOrigin: true,//允许跨域
            pathRewrite: {
                '^/api': ''//请求的时候使用这个api就可以
                // 如：你需要访问：https://cms-api.csdn.net/v1/web_home/select_content?componentIds=www-recomend-community
                // 现在只需这样子输入url： /api/v1/web_home/select_content?componentIds=www-recomend-community  即可
                // 它会变成：http://localhost:8080/api/v1/web_home/select_content?componentIds=www-recomend-community  并允许跨域
            }
        }

    }
  }
})
