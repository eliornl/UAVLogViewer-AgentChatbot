// Worker.js
// import MavlinkParser from 'mavlinkParser'
const mavparser = require('./mavlinkParser')
const DataflashParser = require('./JsDataflashParser/parser').default
const DjiParser = require('./djiParser').default

let parser
self.addEventListener('message', async function (event) {
    if (event.data === null) {
        console.log('WORKER: got bad file message!')
    } else if (event.data.action === 'parse') {
        const data = event.data.file
        console.log('WORKER: Parse action received. isTlog:', event.data.isTlog, 'isDji:', event.data.isDji);
        if (event.data.isTlog) {
            console.log('WORKER: Initializing MavlinkParser');
            parser = new mavparser.MavlinkParser()
            parser.processData(data)
        } else if (event.data.isDji) {
            console.log('WORKER: Initializing DjiParser');
            parser = new DjiParser()
            await parser.processData(data)
        } else {
            console.log('WORKER: Initializing DataflashParser');
            parser = new DataflashParser(true)
            parser.processData(data, ['CMD', 'MSG', 'FILE', 'MODE', 'AHR2', 'ATT', 'GPS', 'POS',
                'XKQ1', 'XKQ', 'NKQ1', 'NKQ2', 'XKQ2', 'PARM', 'MSG', 'STAT', 'EV', 'XKF4', 'FNCE'])
        }

    } else if (event.data.action === 'loadType') {
        if (!parser) {
            console.log('parser not ready')
        }
        parser.loadType(event.data.type.split('[')[0])
    } else if (event.data.action === 'trimFile') {
        parser.trimFile(event.data.time)
    }
})
